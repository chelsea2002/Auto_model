import torch
import json
from FlagEmbedding import FlagModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm  # 引入 tqdm 用于显示进度条


# 1. 处理数据集类
# 1. 数据集定义（直接使用原始文本和SQL）
class TextSQLDataset(Dataset):
    def __init__(self, data, flag_model):  # 修正为 __init__
        # 在初始化时预计算所有嵌入，并显示进度
        self.flag_model = flag_model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        self.texts = []
        self.sqls = []
        self.text_embeds = []
        self.sql_embeds = []
        
        print("Processing dataset...")
        for item in tqdm(data, desc="Encoding texts and SQLs"):
            text = item['question']
            sql = item['query']
            
            # 使用 FlagModel 获取嵌入
            text_embed = self.flag_model.encode(text)  # [1024]
            sql_embed = self.flag_model.encode(sql)    # [1024]
            
            # 存储原始文本、SQL 和对应的嵌入
            self.texts.append(text)
            self.sqls.append(sql)
            self.text_embeds.append(torch.tensor(text_embed, dtype=torch.float32))
            self.sql_embeds.append(torch.tensor(sql_embed, dtype=torch.float32))
    
    def __len__(self):  # 修正为 __len__
        return len(self.texts)
    
    def __getitem__(self, idx):  # 修正为 __getitem__
        # 直接返回预计算的嵌入
        return {
            'text_embed': self.text_embeds[idx],
            'sql_embed': self.sql_embeds[idx]
        }


# TransformerFusion 模型
class TransformerFusion(nn.Module):
    def __init__(self, dim, num_layers=2, heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.alpha = nn.Parameter(torch.tensor(0.7, requires_grad=True))
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.sql_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        # 非线性投影头
        self.proj_sql = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),         # 添加非线性激活
            nn.Dropout(0.1),   # 可选：防止过拟合
            nn.Linear(dim, dim)
        )
        self.proj_text = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),         # 添加非线性激活
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
    def forward(self, v_text, v_sql):
        combined = torch.cat([v_text.unsqueeze(0), v_sql.unsqueeze(0)], dim=0)  # [2, batch, dim]
        output = self.encoder(combined)  # [2, batch, dim]
        enhanced_sql = self.sql_ffn(output[1])  # [batch, dim]
        fusion_embeds = self.alpha * enhanced_sql + (1 - self.alpha) * output[0]  # [batch, dim]
        sql_pred = self.proj_sql(fusion_embeds)
        text_pred = self.proj_text(fusion_embeds)
        return fusion_embeds, v_text, v_sql,sql_pred, text_pred

# 3. 损失函数
def unsupervised_contrastive_loss(fusion_embeds, text_embeds, sql_embeds,sql_pred, text_pred, temperature=0.07, weight_sql=1.0, weight_text=0.5):
    batch_size = fusion_embeds.size(0)
    
    # 归一化嵌入
    fusion_embeds = F.normalize(fusion_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    sql_embeds = F.normalize(sql_embeds, dim=-1)
    
    # 1. 对比损失
    # 计算相似度矩阵 (batch_size x batch_size)
    sim_matrix = torch.matmul(fusion_embeds, fusion_embeds.T) / temperature
    
    # 标签：对角线为正样本，其余为负样本
    labels = torch.arange(batch_size, device=fusion_embeds.device)
    contrastive_loss = F.cross_entropy(sim_matrix, labels)
    
    # 计算重构损失（MSE）
    sql_recon_loss = F.mse_loss(sql_pred, sql_embeds)
    text_recon_loss = F.mse_loss(text_pred, text_embeds)
    
    # 总损失
    total_loss = contrastive_loss + weight_sql * sql_recon_loss + weight_text * text_recon_loss
    
    return total_loss

# 使用示例
# fusion_embeds: [batch_size, d_fusion]，通过注意力机制生成的融合向量
# text_embeds: [batch_size, d_text]，原始文本向量
# sql_embeds: [batch_size, d_sql]，原始SQL向量
# loss = unsupervised_contrastive_loss(fusion_embeds, text_embeds, sql_embeds)


# 4. 训练模型
# 4. 训练函数（添加模型保存）
def train_model(model, train_loader, val_loader, num_epochs=15, lr=5e-5, device='cuda', save_path='best_model.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # 学习率调度
    model.to(device)
    
    best_val_loss = float('inf')
    patience = 3
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            text_embeds = batch['text_embed'].to(device)
            sql_embeds = batch['sql_embed'].to(device)
            
            optimizer.zero_grad()
            fusion_embeds, text_embeds, sql_embeds, sql_pred, text_pred = model(text_embeds, sql_embeds)
                
            loss = unsupervised_contrastive_loss(fusion_embeds, text_embeds, sql_embeds, sql_pred, text_pred)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                text_embeds = batch['text_embed'].to(device)
                sql_embeds = batch['sql_embed'].to(device)
                
                fusion_embeds, text_embeds, sql_embeds, sql_pred, text_pred = model(text_embeds, sql_embeds)
                
                loss = unsupervised_contrastive_loss(fusion_embeds, text_embeds, sql_embeds, sql_pred, text_pred)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型并检查早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Loss: {best_val_loss:.4f} at {save_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break
        
        scheduler.step()  # 更新学习率
    
    final_save_path = save_path.replace('.pth', '_final.pth')
    torch.save(model.state_dict(), final_save_path)
    print(f"Saved final model at {final_save_path}")


# 5. 主程序
if __name__ == "__main__":
    flag_model = FlagModel('C:\\Users\\Administrator\\Desktop\\Codes\\Auto-prompt-model\\plm\\bge-large-en-v1.5', use_fp16=True)
    train_file_name = 'datasets/train.json'
    test_file_name = 'datasets/val.json'
    # 打开并读取 JSON 文件
    with open(train_file_name, "r", encoding="utf-8") as file:
        train_data = json.load(file)  
    with open(test_file_name, "r", encoding="utf-8") as file:
        val_data = json.load(file)  

    # 数据加载
    train_dataset = TextSQLDataset(train_data, flag_model)
    val_dataset = TextSQLDataset(val_data, flag_model)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    # 模型和训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerFusion(dim=1024, num_layers=2, heads=4)
    # 指定保存路径
    save_path = 'C:\\Users\\Administrator\\Desktop\\Codes\\Auto-prompt-model\\models\\best_model.pth'
    train_model(model, train_loader, val_loader, num_epochs=10, save_path=save_path)
