# 已根据新的 Dataset-gemini 布局完成以下两步计划：
# ① 将 fake-all 与 real-all 中的四类图片按 7:3 拆分并重新摆放到 train/<fake|real> 与 val/<fake|real>；
# ② 调整 train.py 中的训练/验证路径指向这些新目录。

# ✂️ 数据拆分情况
# 在仓库根目录运行了一个 Python 脚本，对 fake-all 和 real-all 做分层抽样。
# 切分规则：同一类别内打乱后取约 70% 进 train/<label>，余下放入 val/<label>。
# 拆分脚本也会在每次执行前清空 Dataset-gemini/train/* 与 Dataset-gemini/val/*，
# 避免旧文件残留。

# 注意同步更新train中的文件路径
# val_data_root 现在指向 ["Dataset-gemini/val"]（因为该目录下已经包含 real/ 与 fake/ 子目录）。
# real_folders → ["Dataset-gemini/train/real"]
# fake_folders → ["Dataset-gemini/train/fake"]
# 这样 create_dataloader 与验证循环都能直接使用新的数据位置。文件修改已保存到 train.py
import os
import random
import shutil
from pathlib import Path

BASE = Path('Datasets-4000')
categories = [
    "出入境管理",
    "工商管理",
    "户籍管理",
    "签证",
    "身份证",
    "行驶证",
    "驾驶证",
]
random.seed(418)

source_dirs = {
    'fake': BASE / 'fake-all',
    'real': BASE / 'real-all',
}

dest_dirs = {
    ('train', 'fake'): BASE / 'train' / 'fake',
    ('train', 'real'): BASE / 'train' / 'real',
    ('val', 'fake'): BASE / 'val' / 'fake',
    ('val', 'real'): BASE / 'val' / 'real',
}

# Ensure destination directories exist and are empty
for dest in dest_dirs.values():
    dest.mkdir(parents=True, exist_ok=True)
    for item in dest.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

summary = []

for label, src_dir in source_dirs.items():
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        continue

    for category in categories:
        exts = {'.png', '.jpg', '.jpeg'}
        files = sorted([
            p for p in src_dir.iterdir()
            if p.is_file()
            and p.name.lower().startswith(category.lower())
            and p.suffix.lower() in exts
        ])
        count = len(files)
        if count == 0:
            print(f"No files for category '{category}' in {label}-all")
            continue

        random.shuffle(files)
        if count == 1:
            train_count = 1  # only training to avoid empty
        else:
            train_count = max(1, int(count * 0.7))
            if train_count == count:
                train_count -= 1

        train_dest = dest_dirs[( 'train', label )]
        val_dest = dest_dirs[( 'val', label )]

        for idx, file_path in enumerate(files):
            target_dir = train_dest if idx < train_count else val_dest
            # copy files instead of moving so originals in fake-all/real-all are preserved
            shutil.copy2(str(file_path), target_dir / file_path.name)

        summary.append({
            'label': label,
            'category': category,
            'total': count,
            'train': train_count,
            'val': count - train_count,
        })

# Print summary
print("\nSplit summary:")
for record in summary:
    print(f"{record['label']:4s} | {record['category']}: total={record['total']} -> train={record['train']}, val={record['val']}")
