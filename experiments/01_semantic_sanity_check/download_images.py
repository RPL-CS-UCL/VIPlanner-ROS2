"""Download wild-scene images grouped by category."""
import os
import urllib.request

# 按场景分组
IMAGES = {
    "hallway": [
        "https://images.unsplash.com/photo-1702675299379-1a584c7fe35f?w=600&auto=format&fit=crop&q=60",
        "https://images.unsplash.com/photo-1573247257746-e445228fe991?w=600&auto=format&fit=crop&q=60",
        "https://images.unsplash.com/photo-1504297050568-910d24c426d3?w=600&auto=format&fit=crop&q=60",
    ],
    "urban_sidewalk": [
        "https://images.unsplash.com/photo-1653701888795-a23335d9aee2?w=600&auto=format&fit=crop&q=60",
        "https://images.unsplash.com/photo-1548761434-f8021750c1bd?w=600&auto=format&fit=crop&q=60",
        "https://images.unsplash.com/photo-1638696944009-7cad85589e09?w=600&auto=format&fit=crop&q=60",
    ],
    "forest_trail": [
        "https://images.unsplash.com/photo-1720760585814-5c5280ea72d5?w=600&auto=format&fit=crop&q=60",
        "https://plus.unsplash.com/premium_photo-1665311552973-53cdaeaa52c3?w=600&auto=format&fit=crop&q=60",
        "https://plus.unsplash.com/premium_photo-1665311551953-5ee597cb9d47?w=600&auto=format&fit=crop&q=60",
        "https://images.unsplash.com/photo-1720322117823-885d189427f4?w=600&auto=format&fit=crop&q=60",
    ],
    "mountain_path": [
        "https://plus.unsplash.com/premium_photo-1677396921317-5398bbe55a68?w=600&auto=format&fit=crop&q=60",
        "https://media.istockphoto.com/id/1340528257/photo/woman-hikes-along-trail-in-banff-national-park.webp?a=1&b=1&s=612x612",
        "https://plus.unsplash.com/premium_photo-1676827547827-b011b7da4aed?w=600&auto=format&fit=crop&q=60",
    ],
    "alpine_scree": [
        "https://media.istockphoto.com/id/1446263930/photo/scree-of-white-stones-in-the-middle-of-the-italian-alps-in-summer.webp?a=1&b=1&s=612x612",
    ],
    "snowy_path": [
        "https://images.unsplash.com/photo-1616762715594-81777b216346?w=600&auto=format&fit=crop&q=60",
        "https://images.unsplash.com/photo-1674207899284-33c5bd016c98?w=600&auto=format&fit=crop&q=60",
        "https://plus.unsplash.com/premium_photo-1674161469776-c6f0bac8fd85?w=600&auto=format&fit=crop&q=60",
    ],
    "grass_field": [
        "https://images.unsplash.com/photo-1584623572201-d0385667e46d?w=600&auto=format&fit=crop&q=60",
        "https://images.unsplash.com/photo-1583227498947-cabbd5660987?w=600&auto=format&fit=crop&q=60",
        "https://images.unsplash.com/photo-1592932323118-926a2bf915a6?w=600&auto=format&fit=crop&q=60",
    ],
    "leaf_path": [
        "https://images.unsplash.com/photo-1762756392828-592edfe67cfe?w=600&auto=format&fit=crop&q=60",
        "https://images.unsplash.com/photo-1764131746476-681b22797fc4?w=600&auto=format&fit=crop&q=60",
        "https://plus.unsplash.com/premium_photo-1667511508039-845de3303536?w=600&auto=format&fit=crop&q=60",
    ],
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加 User-Agent 避免某些图站 403
opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

total = 0
for category, urls in IMAGES.items():
    for i, url in enumerate(urls, start=1):
        name = f"{category}_{i:02d}"
        path = os.path.join(OUTPUT_DIR, f"{name}.jpg")
        total += 1

        if os.path.exists(path):
            print(f"  skip: {name}")
            continue

        try:
            urllib.request.urlretrieve(url, path)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

print(f"\nDone. {total} URLs in {OUTPUT_DIR}/")