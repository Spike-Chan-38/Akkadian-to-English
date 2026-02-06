# Deep Past Challenge: Translating Old Assyrian Cuneiform

This repository contains neural machine-translation models for **Old Assyrian**, an early form of Akkadian, aimed at translating transliterated cuneiform tablets into English.  

## Challenge Context
- Old Assyrian texts document Bronze Age trade networks between Mesopotamia and Anatolia.  
- ~23,000 tablets survive; only half have been translated.  
- Texts include letters, invoices, and contracts—the everyday voices of ancient merchants.  
- Old Assyrian is low-resource and morphologically complex, making standard NMT architectures insufficient.  

## Goal
Build machine-translation models that convert transliterated Akkadian into English, giving voice to 10,000+ untranslated tablets and creating a blueprint for other endangered or ancient languages.  

## Contents
- `data/` – Transliteration datasets  
- `models/` – NMT model architectures and checkpoints  
- `notebooks/` – Experiments and evaluation scripts  
- `scripts/` – Training and inference utilities  

## Usage
1. Prepare transliterated Old Assyrian data in `data/`.  
2. Train models with `scripts/train.py`.  
3. Translate tablets with `scripts/translate.py`.  
4. Evaluate outputs using `notebooks/evaluate.ipynb`.  

## License
[MIT License](LICENSE)

