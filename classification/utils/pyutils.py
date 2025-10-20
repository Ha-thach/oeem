import numpy as np 

def chunks(lst, num_workers=None, n=None):
    """
    Helper function: chia list thành các nhóm nhỏ (an toàn cho num_workers=0).
    """
    if not lst:
        return []  # tránh lỗi nếu danh sách trống

    chunk_list = []
    if num_workers is None and n is None:
        raise ValueError("⚠️ chunks() cần ít nhất num_workers hoặc n")

    if n is None:
        # ✅ tránh chia cho 0
        num_workers = max(1, num_workers or 1)
        n = int(np.ceil(len(lst) / num_workers))

    for i in range(0, len(lst), n):
        chunk_list.append(lst[i:i + n])
    return chunk_list
