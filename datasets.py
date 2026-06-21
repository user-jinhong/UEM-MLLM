import os
from PIL import Image
import random
import json
import re
from torch.utils.data import Dataset
from lavis.processors.blip_processors import BlipCaptionProcessor

class CLEVR_Dataset(Dataset):
    def __init__(self, data_path, transform, split='train', max_words=40, prompt="", k=5, rag_store_path=None):
        super().__init__()
        self.data_path = data_path
        self.default_image_path = os.path.join(self.data_path, 'images')
        self.nsc_image_path = os.path.join(self.data_path, 'nsc_images')
        self.sc_image_path = os.path.join(self.data_path, 'sc_images')
        self.split = split
        self.transform = transform
        self.max_words = max_words
        self.prompt = prompt
        self.text_process = BlipCaptionProcessor(prompt=prompt, max_words=max_words)
        self.k = k

        with open(os.path.join(self.data_path, "splits.json"), 'r') as fp:
            total_image_ids = json.load(fp)
            self.image_ids = total_image_ids[split]

        with open(os.path.join(self.data_path, "change_captions.json"), 'r') as fp:
            self.change_captions = json.load(fp)

        with open(os.path.join(self.data_path, "no_change_captions.json"), 'r') as fp:
            self.no_change_captions = json.load(fp)

        self.rag_store = json.load(open(rag_store_path, 'rb')) if rag_store_path else {}

    def __len__(self):
        return len(self.image_ids) * 2

    def __getitem__(self, index):
        base_idx = index // 2
        use_semantic = (index % 2 == 0)  # 偶数：semantic，奇数：no-change

        image_id = self.image_ids[base_idx]
        image_name = f"CLEVR_default_{int(image_id):06d}.png"

        # paths
        bef_image_path = os.path.join(self.default_image_path, image_name)
        sc_image_path = os.path.join(self.sc_image_path, image_name.replace('default', 'semantic'))
        nsc_image_path = os.path.join(self.nsc_image_path, image_name.replace('default', 'nonsemantic'))

        # load ONLY what you need (省IO)
        bef_image = self.get_image_data(bef_image_path)

        if use_semantic:
            aft_image = self.get_image_data(sc_image_path)
            sc_caption = random.choice(self.change_captions[image_name])
            caption = self.text_process(sc_caption)
            image_key = "sc_" + image_name
            img_id = f"{int(image_id):06d}.png"
        else:
            aft_image = self.get_image_data(nsc_image_path)  # no-change 的 after
            nsc_caption = random.choice(self.no_change_captions[image_name])
            caption = self.text_process(nsc_caption)
            image_key = "nsc_" + image_name
            img_id = f"{int(image_id):06d}.png_n"

        lexicon = self.convert_captions_to_lexicon(image_key)

        if self.split == "train":
            # print({"bef_img": bef_image, "aft_img": aft_image, "caption": caption, "lexicon": lexicon})
            return {"bef_img": bef_image, "aft_img": aft_image, "caption": caption, "lexicon": lexicon}
        else:
            return {
                "bef_path": bef_image_path,
                "aft_path": sc_image_path if use_semantic else nsc_image_path,
                "bef_img": bef_image,
                "aft_img": aft_image,
                "img_id": img_id,
                "lexicon": lexicon
            }



    def get_image_data(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def convert_captions_to_lexicon(self, image_id):
        if image_id not in self.rag_store:
            return ""
        sentences = self.rag_store[image_id][:self.k]
        all_words = " ".join(sentences).split(' ')
        filter_words = []
        for w in all_words:
            if w not in filter_words:
                filter_words.append(w)
        return " ".join(filter_words)


    
class SpotDataset(Dataset):
    def __init__(self, image_path, anno_path, racap_path, transform=None, split='train', prompt="", k=5) -> None:
        super().__init__()
        self.image_path = image_path
        self.split = split
        self.transform = transform
        self.text_preprocess = BlipCaptionProcessor(prompt=prompt, max_words=40)
        captions_file_path = os.path.join(anno_path,'filter_{}.json'.format(split))
        with open(captions_file_path,'r') as f:
            self.captions = json.load(f)
        with open(racap_path, 'r') as f:
            self.rag_captions = json.load(f)
        self.k = k

        with open(os.path.join(anno_path, 'filter_train.json'), 'r') as f:
            training_captions = json.load(f)
        self.texts = []
        for cap in training_captions:
            for sentence in cap['sentences']:
                self.texts.append(self.text_preprocess(sentence))
           
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = caption['img_id']
        text = random.choice(caption['sentences'])
        text = self.text_preprocess(text)
        bef_img = self.get_image(img_id)
        aft_img = self.get_image(img_id+'_2')
        
        assert img_id in self.rag_captions.keys(), "image %s does not has retrieval captions" % img_id
        ref_words = self.convert_captions_to_lexicon(self.rag_captions[img_id])
        
        output = {}
        if self.split == 'train':
            output['bef_img'] = bef_img
            output['aft_img'] = aft_img
            output['caption'] = text
            output['lexicon'] = ref_words
        else:
            output['bef_path'] = os.path.join(self.image_path, '%s.png' % img_id)
            output['aft_path'] = os.path.join(self.image_path, '%s.png' % (img_id+"_2"))
            output['bef_img'] = bef_img
            output['aft_img'] = aft_img
            output['img_id'] = "%s.png" % img_id
            output['lexicon'] = ref_words
            
        return output
    
    def get_image(self, img_id):
        img_path = os.path.join(self.image_path, '%s.png' % img_id)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def convert_captions_to_lexicon(self, racaps):
        sentences = racaps[:self.k]
        all_words = " ".join(sentences).split(' ')
        filter_words = []
        for w in all_words:
            if w not in filter_words:
                filter_words.append(w)
        return " ".join(filter_words)
    
class Image_Edit_Request(Dataset):  
    def __init__(self, data_path, transform, split="split", max_words=40, prompt="", rag_store_path=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.image_path = os.path.join(data_path, "images")
        self.split = split
        self.transform = transform
        self.max_words = max_words
        self.prompt = prompt
        self.text_process = BlipCaptionProcessor(prompt=prompt, max_words=max_words)
        self.word_list = None
        self.rag_captions = None
        if rag_store_path:
            with open(rag_store_path, "r") as f:
                self.rag_captions = json.load(f)

        with open(os.path.join(self.data_path, "%s.json"%self.split), "r") as f:
            self.captions = json.load(f)

    def __len__(self,):
        return len(self.captions)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        text = random.choice(caption['sents'])
        text = self.text_process(text)
        bef_img = self.get_image(caption['img0'])
        aft_img = self.get_image(caption['img1'])
        img_id = caption['uid']
        if self.rag_captions is not None:
            ref_words = self.convert_captions_to_lexicon(self.rag_captions[img_id], k=5)
        else:
            ref_words = None
        out = {}
        if self.split == 'train':
            out['bef_img'] = bef_img
            out['aft_img'] = aft_img
            out['caption'] = text
            out['lexicon'] = ref_words if ref_words else "1"
        else:
            out['bef_img'] = bef_img
            out['aft_img'] = aft_img
            out['img_id'] = img_id
            out['lexicon'] = ref_words if ref_words else "1"
            out['bef_path'] = os.path.join(self.image_path, caption['img0'])
            out['aft_path'] = os.path.join(self.image_path, caption['img1'])
        return out
    
    def get_image(self, image_name):
        img_path = os.path.join(self.image_path, image_name)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def convert_captions_to_lexicon(self, racaps, k=3):
        all_words = " ".join(racaps[:k]).split(' ')
        filter_words = []
        for w in all_words:
            if w not in filter_words:
                filter_words.append(w)
        return " ".join(filter_words)
    

class BirdDataset(Dataset):
    def __init__(self, image_dir, data_root, transform=None, split='train', prompt="", max_words=40):
        super().__init__()
        self.image_dir = image_dir
        self.split = split
        self.transform = transform
        self.text_process = BlipCaptionProcessor(prompt=prompt, max_words=max_words)

        anno_path = os.path.join(data_root, f"{split}.json")
        with open(anno_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        try:
            raw_pairs = json.loads(text)
        except json.JSONDecodeError:
            raw_pairs = [json.loads(line) for line in text.splitlines() if line.strip()]

        self.pairs = []

        for item in raw_pairs:
            img1_name = item["img1"]
            img2_name = item["img2"]

            sentences = item.get("sentences", [])

            # 兼容 sentences 是 str 的情况
            if isinstance(sentences, str):
                sentences = [sentences]

            # 兼容 sentences 是 [{"raw": "..."}] 或 [{"caption": "..."}] 的情况
            clean_sentences = []
            for s in sentences:
                if isinstance(s, str):
                    cap = s.strip()
                elif isinstance(s, dict):
                    cap = (
                        s.get("raw")
                        or s.get("caption")
                        or s.get("sentence")
                        or s.get("sent")
                        or ""
                    ).strip()
                else:
                    cap = ""

                if cap:
                    clean_sentences.append(cap)
            self.pairs.append({
                "img1": img1_name,
                "img2": img2_name,
                "captions": clean_sentences,
            })


        print(f"[BirdDataset] split={split}, samples={len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        item = self.pairs[index]

        img1_name = item["img1"]
        img2_name = item["img2"]
        # raw_caption = item["caption"]
        #
        # caption = self.text_process(raw_caption)
        if len(item["captions"]) > 0:
            raw_caption = random.choice(item["captions"])
        else:
            raw_caption = ""

        caption = self.text_process(raw_caption)
        img1 = self.get_image(img1_name)
        img2 = self.get_image(img2_name)

        # lexicon 用所有 caption 的词，而不是只用当前 caption
        # lexicon = self.extract_keywords(" ".join(item.get("all_captions", [raw_caption])))
        lexicon = self.extract_keywords(" ".join(item.get("captions", [raw_caption])))
        if self.split == "train":
            return {
                "bef_img": img1,
                "aft_img": img2,
                "caption": caption,
                "lexicon": lexicon,
            }
        else:
            return {
                "bef_path": os.path.join(self.image_dir, img1_name),
                "aft_path": os.path.join(self.image_dir, img2_name),
                "img_id": f"{img1_name}_{img2_name}",
                "bef_img": img1,
                "aft_img": img2,
                "caption": caption,
                "lexicon": lexicon,
            }

    def get_image(self, filename):
        path = os.path.join(self.image_dir, filename)
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def extract_keywords(self, sentence):
        words = sentence.lower().split()
        unique_words = []
        for w in words:
            if w not in unique_words:
                unique_words.append(w)
        return " ".join(unique_words)


class JsonlEditDataset(Dataset):
    """
    For jsonl like:
      {"image": "...", "edited_image": "...", "change_caption": "..."}
    Returns:
      train:  bef_img, aft_img, caption, lexicon
      eval :  bef_path, aft_path, bef_img, aft_img, img_id, caption, lexicon
    """
    def __init__(
        self,
        jsonl_path,
        transform=None,
        split="train",
        prompt="",
        max_words=40,
        rag_store_path=None,
        k=5,
        drop_if_missing=True,
    ):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.transform = transform
        self.split = split
        self.text_process = BlipCaptionProcessor(prompt=prompt, max_words=max_words)
        self.k = k
        self.drop_if_missing = drop_if_missing

        # Optional: retrieval lexicon store (dict: img_id or key -> list[str])
        self.rag_store = None
        if rag_store_path:
            # rag_store_path 通常是 json
            with open(rag_store_path, "r", encoding="utf-8") as f:
                self.rag_store = json.load(f)

        self.samples = []
        self._load_jsonl()

    def _load_jsonl(self):
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    if self.drop_if_missing:
                        continue
                    raise

                img = item.get("image", None)
                edited = item.get("edited_image", None)
                cap = item.get("change_caption", None)

                if not img or not edited or cap is None:
                    if self.drop_if_missing:
                        continue
                    else:
                        raise ValueError(f"Missing keys at line {line_idx}")

                # 可选：过滤不存在的图片路径，避免训练时报错
                if self.drop_if_missing:
                    if (not os.path.exists(img)) or (not os.path.exists(edited)):
                        continue

                self.samples.append({
                    "image": img,
                    "edited_image": edited,
                    "change_caption": cap
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        bef_path = item["image"]
        aft_path = item["edited_image"]

        # caption：你这里只有一句 change_caption，不需要 random.choice
        caption_raw = item["change_caption"]
        caption = self.text_process(caption_raw)

        bef_img = self.get_image(bef_path)
        aft_img = self.get_image(aft_path)

        # img_id：用文件名更稳
        img_id = os.path.basename(bef_path)

        # lexicon：如果你没有 rag_store，就返回 "1" 兼容你现有代码
        lexicon = "1"
        if self.rag_store is not None:
            # 这里给你两种 key 兼容：用 img_id 或用完整路径
            key = img_id if img_id in self.rag_store else bef_path
            lexicon = self.convert_captions_to_lexicon(key) if key in self.rag_store else "1"

        if self.split == "train":
            return {
                "bef_img": bef_img,
                "aft_img": aft_img,
                "caption": caption,
                "lexicon": lexicon
            }
        else:
            return {
                "bef_path": bef_path,
                "aft_path": aft_path,
                "bef_img": bef_img,
                "aft_img": aft_img,
                "img_id": img_id,
                "caption": caption,
                "lexicon": lexicon
            }

    def get_image(self, path):
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def convert_captions_to_lexicon(self, key):
        # rag_store[key] should be a list of sentences
        sentences = self.rag_store[key][: self.k]
        all_words = " ".join(sentences).split(" ")
        uniq = []
        for w in all_words:
            if w and (w not in uniq):
                uniq.append(w)
        return " ".join(uniq)