#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys

def carica_file(nome_file):
    """Legge un file JSON e restituisce un dizionario con video_name come chiave."""
    try:
        with open(nome_file, 'r', encoding='utf-8') as f:
            dati = json.load(f)
        
        # Mappiamo video_name → category
        mappa = {el["video_name"]: el["category"] for el in dati if "video_name" in el and "category" in el}
        return mappa
    except Exception as e:
        print(f"Errore nel caricamento di {nome_file}: {e}")
        sys.exit(1)

def confronta(file1, file2):
    mappa1 = carica_file(file1)
    mappa2 = carica_file(file2)

    nomi1 = set(mappa1.keys())
    nomi2 = set(mappa2.keys())

    # 1. Videoname presenti in entrambi ma con categorie diverse
    common = nomi1 & nomi2
    print(f"--- DIFFERENZE DI CATEGORIA ({len(common)} video in comune) ---")
    diff_count = 0
    for vid in common:
        cat1 = mappa1[vid]
        cat2 = mappa2[vid]
        if cat1 != cat2:
            print(f"DIFF: '{vid}' -> {cat1} (f1) vs {cat2} (f2)")
            diff_count += 1
    if diff_count == 0:
        print("Nessuna differenza di categoria trovata nei video comuni.")

    print("\n" + "-"*50 + "\n")

    # 2. Videoname presenti SOLO nel FILE 1
    only_in_1 = nomi1 - nomi2
    print(f"--- SOLO IN {file1} ({len(only_in_1)} video) ---")
    for vid in sorted(only_in_1):
        print(f"FILE1 ONLY: '{vid}' [Cat: {mappa1[vid]}]")

    print("\n" + "-"*50 + "\n")

    # 3. Videoname presenti SOLO nel FILE 2
    only_in_2 = nomi2 - nomi1
    print(f"--- SOLO IN {file2} ({len(only_in_2)} video) ---")
    for vid in sorted(only_in_2):
        print(f"FILE2 ONLY: '{vid}' [Cat: {mappa2[vid]}]")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python script.py <file_json1> <file_json2>")
        sys.exit(1)

    confronta(sys.argv[1], sys.argv[2])