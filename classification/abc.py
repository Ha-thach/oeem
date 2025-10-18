from torch.utils.data import DataLoader
import dataset

# ====== Cấu hình ======
data_path = '/Users/thachha/Desktop/AIO2025-official/AIMA/CP- WSSS/PBIP/data/BCSS-WSSS/sub/train'
batch_size = 4
num_class = 5

# ====== Dataset ======
ds = dataset.BCSS_WSSS_Dataset(data_path_name=data_path, num_class=num_class)
print(f"📁 Dataset initialized — total samples found: {len(ds)}")

# ====== DataLoader ======
loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

# ====== Đếm số batch và tổng số mẫu load được ======
total_samples = 0
for i, (imgs, labels) in enumerate(loader):
    total_samples += imgs.size(0)
    print(f"🔹 Batch {i+1}: {imgs.size(0)} samples, tensor shape = {imgs.shape}")

print(f"\n✅ Tổng số batch: {len(loader)}")
print(f"✅ Tổng số mẫu load được từ DataLoader: {total_samples}")
