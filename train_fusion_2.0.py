import torch
import json
import os
from FlagEmbedding import FlagModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 1. 数据集类
class TextSQLDataset(Dataset):
    def __init__(self, embeddings_file=None, data=None, flag_model=None):
        if embeddings_file and os.path.exists(embeddings_file):
            # Load precomputed embeddings
            print(f"Loading precomputed embeddings from {embeddings_file}...")
            embeddings = torch.load(embeddings_file)
            self.main_texts = embeddings['main_texts']
            self.main_sqls = embeddings['main_sqls']
            self.main_text_embeds = embeddings['main_text_embeds']
            self.main_sql_embeds = embeddings['main_sql_embeds']
            self.similar_texts = embeddings['similar_texts']
            self.similar_sqls = embeddings['similar_sqls']
            self.similar_text_embeds = embeddings['similar_text_embeds']
            self.similar_sql_embeds = embeddings['similar_sql_embeds']
        elif data and flag_model:
            # Compute embeddings on-the-fly (for preprocessing)
            self.flag_model = flag_model
            self.main_texts = []
            self.main_sqls = []
            self.main_text_embeds = []
            self.main_sql_embeds = []
            self.similar_texts = []
            self.similar_sqls = []
            self.similar_text_embeds = []
            self.similar_sql_embeds = []
            
            print("Processing dataset...")
            for item in tqdm(data, desc="Encoding texts and SQLs"):
                main_text = item['question']
                main_sql = item['query']
                main_text_embed = self.flag_model.encode(main_text)
                main_sql_embed = self.flag_model.encode(main_sql)
                
                self.main_texts.append(main_text)
                self.main_sqls.append(main_sql)
                self.main_text_embeds.append(torch.tensor(main_text_embed, dtype=torch.float32))
                self.main_sql_embeds.append(torch.tensor(main_sql_embed, dtype=torch.float32))
                
                similar_texts = []
                similar_sqls = []
                similar_text_embeds = []
                similar_sql_embeds = []
                for sim_item in item['similar']:
                    sim_text = sim_item['question']
                    sim_sql = sim_item['query']
                    sim_text_embed = self.flag_model.encode(sim_text)
                    sim_sql_embed = self.flag_model.encode(sim_sql)
                    
                    similar_texts.append(sim_text)
                    similar_sqls.append(sim_sql)
                    similar_text_embeds.append(torch.tensor(sim_text_embed, dtype=torch.float32))
                    similar_sql_embeds.append(torch.tensor(sim_sql_embed, dtype=torch.float32))
                
                self.similar_texts.append(similar_texts)
                self.similar_sqls.append(similar_sqls)
                self.similar_text_embeds.append(torch.stack(similar_text_embeds))  # [3, dim]
                self.similar_sql_embeds.append(torch.stack(similar_sql_embeds))   # [3, dim]
        else:
            raise ValueError("Must provide either embeddings_file or both data and flag_model")
    
    def __len__(self):
        return len(self.main_texts)
    
    def __getitem__(self, idx):
        return {
            'main_text_embed': self.main_text_embeds[idx],        # [dim]
            'main_sql_embed': self.main_sql_embeds[idx],          # [dim]
            'similar_text_embeds': self.similar_text_embeds[idx], # [3, dim]
            'similar_sql_embeds': self.similar_sql_embeds[idx]    # [3, dim]
        }

    def save_embeddings(self, save_path):
        embeddings = {
            'main_texts': self.main_texts,
            'main_sqls': self.main_sqls,
            'main_text_embeds': self.main_text_embeds,
            'main_sql_embeds': self.main_sql_embeds,
            'similar_texts': self.similar_texts,
            'similar_sqls': self.similar_sqls,
            'similar_text_embeds': self.similar_text_embeds,
            'similar_sql_embeds': self.similar_sql_embeds
        }
        torch.save(embeddings, save_path)
        print(f"Embeddings saved to {save_path}")

# 2. TransformerFusion 模型
class TransformerFusion(nn.Module):
    def __init__(self, dim, num_layers=6, heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.alpha = nn.Parameter(torch.tensor(0.7, requires_grad=True))
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.sql_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )
        self.proj_sql = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )
        self.proj_text = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )
    
    def forward(self, v_text, v_sql):
        combined = torch.cat([v_text.unsqueeze(0), v_sql.unsqueeze(0)], dim=0)  # [2, batch, dim]
        output = self.encoder(combined)  # [2, batch, dim]
        enhanced_sql = self.sql_ffn(output[1])  # [batch, dim]
        fusion_embeds = self.alpha * enhanced_sql + (1 - self.alpha) * output[0]  # [batch, dim]
        sql_pred = self.proj_sql(fusion_embeds)
        text_pred = self.proj_text(fusion_embeds)
        return fusion_embeds, v_text, v_sql, sql_pred, text_pred

# 3. 损失函数
def custom_loss(main_fusion, similar_fusions, text_embeds, sql_embeds, sql_pred, text_pred, temperature=0.07, weight_sql=1.0, weight_text=0.5, weight_sim=5.0):
    batch_size = main_fusion.size(0)
    
    # 归一化嵌入
    main_fusion = F.normalize(main_fusion, dim=-1)
    similar_fusions = [F.normalize(sim, dim=-1) for sim in similar_fusions]  # 列表，每个为 [batch, dim]
    text_embeds = F.normalize(text_embeds, dim=-1)
    sql_embeds = F.normalize(sql_embeds, dim=-1)
    
    # 对比损失（主对之间）
    sim_matrix = torch.matmul(main_fusion, main_fusion.T) / temperature
    labels = torch.arange(batch_size, device=main_fusion.device)
    contrastive_loss = F.cross_entropy(sim_matrix, labels)
    
    # 重构损失（仅对主对计算）
    sql_recon_loss = F.mse_loss(sql_pred, sql_embeds)
    text_recon_loss = F.mse_loss(text_pred, text_embeds)
    
    # 相似度损失（主对与相似对）
    similarity_loss = 0.0
    for sim_fusion in similar_fusions:
        sim = F.cosine_similarity(main_fusion, sim_fusion)
        similarity_loss += (1 - sim.mean())
    similarity_loss /= len(similar_fusions)  # 平均3个相似对
    
    # 总损失
    total_loss = contrastive_loss + weight_sql * sql_recon_loss + weight_text * text_recon_loss + weight_sim * similarity_loss
    return total_loss

# 4. 训练函数
def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4, device='cuda', save_path='best_model.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.to(device)
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            main_text_embeds = batch['main_text_embed'].to(device)  # [batch, dim]
            main_sql_embeds = batch['main_sql_embed'].to(device)    # [batch, dim]
            similar_text_embeds = batch['similar_text_embeds']      # [batch, 3, dim]
            similar_sql_embeds = batch['similar_sql_embeds']        # [batch, 3, dim]
            
            optimizer.zero_grad()
            
            # 主对的融合嵌入
            main_fusion, _, _, main_sql_pred, main_text_pred = model(main_text_embeds, main_sql_embeds)
            
            # 相似对的融合嵌入
            similar_fusions = []
            for i in range(3):
                sim_text = similar_text_embeds[:, i, :].to(device)  # [batch, dim]
                sim_sql = similar_sql_embeds[:, i, :].to(device)    # [batch, dim]
                sim_fusion, _, _, _, _ = model(sim_text, sim_sql)
                similar_fusions.append(sim_fusion)
            
            # 计算损失
            loss = custom_loss(main_fusion, similar_fusions, main_text_embeds, main_sql_embeds, main_sql_pred, main_text_pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                main_text_embeds = batch['main_text_embed'].to(device)
                main_sql_embeds = batch['main_sql_embed'].to(device)
                similar_text_embeds = batch['similar_text_embeds']
                similar_sql_embeds = batch['similar_sql_embeds']
                
                main_fusion, _, _, main_sql_pred, main_text_pred = model(main_text_embeds, main_sql_embeds)
                similar_fusions = []
                for i in range(3):
                    sim_text = similar_text_embeds[:, i, :].to(device)
                    sim_sql = similar_sql_embeds[:, i, :].to(device)
                    sim_fusion, _, _, _, _ = model(sim_text, sim_sql)
                    similar_fusions.append(sim_fusion)
                
                loss = custom_loss(main_fusion, similar_fusions, main_text_embeds, main_sql_embeds, main_sql_pred, main_text_pred)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Loss: {best_val_loss:.4f} at {save_path}")
        
        scheduler.step()
    
    final_save_path = save_path.replace('.pth', '_final.pth')
    torch.save(model.state_dict(), final_save_path)
    print(f"Saved final model at {final_save_path}")

# 5. 主程序
if __name__ == "__main__":
    # Paths
    embeddings_file = 'embeddings/embeddings.pt'
    data_file = 'datasets/train.json'
    save_path = 'C:\\Users\\Administrator\\Desktop\\Codes\\Auto-prompt-model\\models\\best_model.pth'
    
    # Load data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if embeddings exist; if not, preprocess and save them
    if not os.path.exists(embeddings_file):
        flag_model = FlagModel('C:\\Users\\Administrator\\Desktop\\Codes\\Auto-prompt-model\\plm\\bge-large-en-v1.5', use_fp16=True)
        full_dataset = TextSQLDataset(data=data, flag_model=flag_model)
        os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
        full_dataset.save_embeddings(embeddings_file)
    else: 
        full_dataset = TextSQLDataset(embeddings_file=embeddings_file)
    
    # Split dataset
    train_size = int(0.9 * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, len(full_dataset) - train_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerFusion(dim=1024, num_layers=6, heads=8)
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4, device=device, save_path=save_path)
