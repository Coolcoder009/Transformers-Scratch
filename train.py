# import os
# from transformer import Transformer
# from transformer import create_padding_mask
# from transformer import create_causal_mask
# from transformer import combine_masks
# import torch
# import torch.nn as nn
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("using:",{device})
# os.makedirs("output", exist_ok=True)
#
# START_TOKEN = '<SOS>'
# PADDING_TOKEN = '<PAD>'
# END_TOKEN = '<EOS>'
# UNKNOWN_TOKEN = '<UNK>'
# ta_vocab = [PADDING_TOKEN, START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
#             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', 'ˌ',
#             'ஃ', 'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ', 'க்', 'க', 'கா', 'கி', 'கீ', 'கு', 'கூ',
#             'கெ',
#             'கே', 'கை', 'கொ', 'கோ', 'கௌ', 'ங்', 'ங', 'ஙா', 'ஙி', 'ஙீ', 'ஙு', 'ஙூ', 'ஙெ', 'ஙே', 'ஙை', 'ஙொ', 'ஙோ', 'ஙௌ',
#             'ச்',
#             'ச', 'சா', 'சி', 'சீ', 'சு', 'சூ', 'செ', 'சே', 'சை', 'சொ', 'சோ', 'சௌ',
#             'ஞ்', 'ஞ', 'ஞா', 'ஞி', 'ஞீ', 'ஞு', 'ஞூ', 'ஞெ', 'ஞே', 'ஞை', 'ஞொ', 'ஞோ', 'ஞௌ',
#             'ட்', 'ட', 'டா', 'டி', 'டீ', 'டு', 'டூ', 'டெ', 'டே', 'டை', 'டொ', 'டோ', 'டௌ',
#             'ண்', 'ண', 'ணா', 'ணி', 'ணீ', 'ணு', 'ணூ', 'ணெ', 'ணே', 'ணை', 'ணொ', 'ணோ', 'ணௌ',
#             'த்', 'த', 'தா', 'தி', 'தீ', 'து', 'தூ', 'தெ', 'தே', 'தை', 'தொ', 'தோ', 'தௌ',
#             'ந்', 'ந', 'நா', 'நி', 'நீ', 'நு', 'நூ', 'நெ', 'நே', 'நை', 'நொ', 'நோ', 'நௌ',
#             'ப்', 'ப', 'பா', 'பி', 'பீ', 'பு', 'பூ', 'பெ', 'பே', 'பை', 'பொ', 'போ', 'பௌ',
#             'ம்', 'ம', 'மா', 'மி', 'மீ', 'மு', 'மூ', 'மெ', 'மே', 'மை', 'மொ', 'மோ', 'மௌ',
#             'ய்', 'ய', 'யா', 'யி', 'யீ', 'யு', 'யூ', 'யெ', 'யே', 'யை', 'யொ', 'யோ', 'யௌ',
#             'ர்', 'ர', 'ரா', 'ரி', 'ரீ', 'ரு', 'ரூ', 'ரெ', 'ரே', 'ரை', 'ரொ', 'ரோ', 'ரௌ',
#             'ல்', 'ல', 'லா', 'லி', 'லீ', 'லு', 'லூ', 'லெ', 'லே', 'லை', 'லொ', 'லோ', 'லௌ',
#             'வ்', 'வ', 'வா', 'வி', 'வீ', 'வு', 'வூ', 'வெ', 'வே', 'வை', 'வொ', 'வோ', 'வௌ',
#             'ழ்', 'ழ', 'ழா', 'ழி', 'ழீ', 'ழு', 'ழூ', 'ழெ', 'ழே', 'ழை', 'ழொ', 'ழோ', 'ழௌ',
#             'ள்', 'ள', 'ளா', 'ளி', 'ளீ', 'ளு', 'ளூ', 'ளெ', 'ளே', 'ளை', 'ளொ', 'ளோ', 'ளௌ',
#             'ற்', 'ற', 'றா', 'றி', 'றீ', 'று', 'றூ', 'றெ', 'றே', 'றை', 'றொ', 'றோ', 'றௌ',
#             'ன்', 'ன', 'னா', 'னி', 'னீ', 'னு', 'னூ', 'னெ', 'னே', 'னை',
#             'ஶ்', 'ஶ', 'ஶா', 'ஶி', 'ஶீ', 'ஶு', 'ஶூ', 'ஶெ', 'ஶே', 'ஶை', 'ஶொ', 'ஶோ', 'ஶௌ',
#             'ஜ்', 'ஜ', 'ஜா', 'ஜி', 'ஜீ', 'ஜு', 'ஜூ', 'ஜெ', 'ஜே', 'ஜை', 'ஜொ', 'ஜோ', 'ஜௌ',
#             'ஷ்', 'ஷ', 'ஷா', 'ஷி', 'ஷீ', 'ஷு', 'ஷூ', 'ஷெ', 'ஷே', 'ஷை', 'ஷொ', 'ஷோ', 'ஷௌ',
#             'ஸ்', 'ஸ', 'ஸா', 'ஸி', 'ஸீ', 'ஸு', 'ஸூ', 'ஸெ', 'ஸே', 'ஸை', 'ஸொ', 'ஸோ', 'ஸௌ',
#             'ஹ்', 'ஹ', 'ஹா', 'ஹி', 'ஹீ', 'ஹு', 'ஹூ', 'ஹெ', 'ஹே', 'ஹை', 'ஹொ', 'ஹோ', 'ஹௌ',
#             'க்ஷ்', 'க்ஷ', 'க்ஷா', 'க்ஷ', 'க்ஷீ', 'க்ஷு', 'க்ஷூ', 'க்ஷெ', 'க்ஷே', 'க்ஷை', 'க்ஷொ', 'க்ஷோ', 'க்ஷௌ',
#             '்', 'ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', END_TOKEN]
# en_vocab = [PADDING_TOKEN, START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
#             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#             ':', '<', '=', '>', '?', '@',
#             '[', '\\', ']', '^', '_', '`',
#             'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
#             'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
#             'y', 'z', '{', '|', '}', '~', END_TOKEN]
# index_to_tamil = {k: v for k, v in enumerate(ta_vocab)}
# tamil_to_index = {v: k for k, v in enumerate(ta_vocab)}
# index_to_english = {k: v for k, v in enumerate(en_vocab)}
# english_to_index = {v: k for k, v in enumerate(en_vocab)}
# with open('en-ta/English.txt', 'r',encoding="utf8") as file:
#     en_sentences = file.readlines()
# with open('en-ta/Tamil.txt', 'r',encoding="utf8") as file:
#     ta_sentences = file.readlines()
#
# TOTAL_SENTENCES = 200000
# en_sentences = en_sentences[:TOTAL_SENTENCES]
# ta_sentences = ta_sentences[:TOTAL_SENTENCES]
# en_sentences = [sentence.rstrip('\n').lower() for sentence in en_sentences]
# ta_sentences = [sentence.rstrip('\n') for sentence in ta_sentences]
#
#
# def is_valid_token(sentence, vocab):
#     return all(token in vocab for token in sentence)
#
#
# def find_invalid_tokens(sentence, vocab):
#     return [token for token in set(sentence) if token not in vocab]
#
#
# def is_valid_length(sentence, max_sequence_length):
#     return len(sentence) <= max_sequence_length
#
#
# invalid_tokens_list = []
# valid_sentence_indices = []
# invalid_sentence_indices = []
#
# for index, (ta_sentence, en_sentence) in enumerate(zip(ta_sentences, en_sentences)):
#     invalid_ta_tokens = find_invalid_tokens(ta_sentence, ta_vocab)
#     invalid_en_tokens = find_invalid_tokens(en_sentence, en_vocab)
#
#     if is_valid_length(ta_sentence, 250) and is_valid_length(en_sentence, 250):
#         if is_valid_token(ta_sentence, ta_vocab) and is_valid_token(en_sentence, en_vocab):
#             valid_sentence_indices.append(index)
#         else:
#             invalid_tokens_list.append((invalid_ta_tokens, invalid_en_tokens))
#             invalid_sentence_indices.append(index)
#
# print(f"Number of sentences: {len(ta_sentences)}")
# print(f"Number of valid sentences: {len(valid_sentence_indices)}")
#
# ta_sentences = [ta_sentences[i] for i in valid_sentence_indices]
# en_sentences = [en_sentences[i] for i in valid_sentence_indices]
#
#
# def tokenize_sentence(sentence):
#     return list(sentence)
#
#
# def tokens_to_indices(tokens, vocab_to_index):
#     return [vocab_to_index[token] for token in tokens]
#
#
# def add_special_tokens(indices, sos_token_index, eos_token_index):
#     return [sos_token_index] + indices + [eos_token_index]
#
#
# from torch.nn.utils.rnn import pad_sequence
#
#
# def pad_sequences(batch, padding_value):
#     return pad_sequence(batch, batch_first=True, padding_value=padding_value)
#
#
# from torch.utils.data import Dataset
#
#
# class TranslationDataset(Dataset):
#     def __init__(self, source_sentences, target_sentences,
#                  source_vocab_to_index, target_vocab_to_index,
#                  max_length=250):
#         self.source_sentences = source_sentences
#         self.target_sentences = target_sentences
#         self.source_vocab_to_index = source_vocab_to_index
#         self.target_vocab_to_index = target_vocab_to_index
#         self.max_length = max_length
#
#         self.source_sos = source_vocab_to_index['<SOS>']
#         self.source_eos = source_vocab_to_index['<EOS>']
#         self.source_pad = source_vocab_to_index['<PAD>']
#
#         self.target_sos = target_vocab_to_index['<SOS>']
#         self.target_eos = target_vocab_to_index['<EOS>']
#         self.target_pad = target_vocab_to_index['<PAD>']
#
#     def __len__(self):
#         return len(self.source_sentences)
#
#     def __getitem__(self, idx):
#         # Tokenize sentences
#         src_tokens = tokenize_sentence(self.source_sentences[idx])
#         tgt_tokens = tokenize_sentence(self.target_sentences[idx])
#
#         # Convert tokens to indices
#         src_indices = tokens_to_indices(src_tokens, self.source_vocab_to_index)
#         tgt_indices = tokens_to_indices(tgt_tokens, self.target_vocab_to_index)
#
#         # Add special tokens
#         src_indices = add_special_tokens(src_indices, self.source_sos, self.source_eos)
#         tgt_indices = add_special_tokens(tgt_indices, self.target_sos, self.target_eos)
#
#         # Convert to tensors
#         src_tensor = torch.tensor(src_indices, dtype=torch.long)
#         tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
#
#         return src_tensor, tgt_tensor
#
#
# def collate_fn(batch):
#     src_batch, tgt_batch = zip(*batch)
#     src_batch = pad_sequence(src_batch, batch_first=True, padding_value=english_to_index['<PAD>'])
#     tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tamil_to_index['<PAD>'])
#     return src_batch, tgt_batch
#
#
# from torch.utils.data import DataLoader
#
# dataset = TranslationDataset(
#     source_sentences=en_sentences,
#     target_sentences=ta_sentences,
#     source_vocab_to_index=english_to_index,
#     target_vocab_to_index=tamil_to_index
# )
#
# batch_size = 32
#
# dataloader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=collate_fn
# )
# model = Transformer(
#     num_layers=6,
#     d_model=512,
#     dff=2048,
#     dropout=0.1,
#     heads=8,
#     src_vocab_size=len(en_vocab),
#     tgt_vocab_size=len(ta_vocab),
#     max_len=252
# ).to(device)
# loss_fn = nn.CrossEntropyLoss(ignore_index=tamil_to_index['<PAD>'])
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# num_epochs = 24
#
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for src_batch, tgt_batch in dataloader:
#         src_batch = src_batch.to(device)
#         tgt_batch = tgt_batch.to(device)
#
#         src_padding_mask = create_padding_mask(src_batch, pad_token=english_to_index['<PAD>']).to(device)
#         tgt_padding_mask = create_padding_mask(tgt_batch[:, :-1], pad_token=tamil_to_index['<PAD>']).to(device)
#
#         seq_len = tgt_batch[:, :-1].size(1)
#         causal_mask = create_causal_mask(seq_len).to(device)
#         combined_mask = combine_masks(tgt_padding_mask, causal_mask).to(device)
#
#         output = model(
#             src=src_batch,
#             tgt=tgt_batch[:, :-1],
#             src_padding_mask=src_padding_mask,
#             tgt_padding_mask=None,
#             combined_mask=combined_mask
#         )
#
#         target_output = tgt_batch[:, 1:]
#         loss = loss_fn(output.reshape(-1, output.size(-1)), target_output.reshape(-1))
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     avg_loss = total_loss / len(dataloader)
#     print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
#
#     # Save model at end of each epoch
#     model_path = f"output/transformer_epoch_{epoch + 1}.pt"
#     torch.save(model.state_dict(), model_path)
#     print(f"Saved model to {model_path}")

import os
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer import Transformer, create_padding_mask, create_causal_mask, combine_masks

# === Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)
os.makedirs("output", exist_ok=True)

# === Special Tokens and Vocab ===
START_TOKEN = '<SOS>'
PADDING_TOKEN = '<PAD>'
END_TOKEN = '<EOS>'
UNKNOWN_TOKEN = '<UNK>'

# Define vocab lists (ta_vocab and en_vocab) here — omitted for brevity
ta_vocab = [PADDING_TOKEN, START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', 'ˌ',
            'ஃ', 'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ', 'க்', 'க', 'கா', 'கி', 'கீ', 'கு', 'கூ',
            'கெ',
            'கே', 'கை', 'கொ', 'கோ', 'கௌ', 'ங்', 'ங', 'ஙா', 'ஙி', 'ஙீ', 'ஙு', 'ஙூ', 'ஙெ', 'ஙே', 'ஙை', 'ஙொ', 'ஙோ', 'ஙௌ',
            'ச்',
            'ச', 'சா', 'சி', 'சீ', 'சு', 'சூ', 'செ', 'சே', 'சை', 'சொ', 'சோ', 'சௌ',
            'ஞ்', 'ஞ', 'ஞா', 'ஞி', 'ஞீ', 'ஞு', 'ஞூ', 'ஞெ', 'ஞே', 'ஞை', 'ஞொ', 'ஞோ', 'ஞௌ',
            'ட்', 'ட', 'டா', 'டி', 'டீ', 'டு', 'டூ', 'டெ', 'டே', 'டை', 'டொ', 'டோ', 'டௌ',
            'ண்', 'ண', 'ணா', 'ணி', 'ணீ', 'ணு', 'ணூ', 'ணெ', 'ணே', 'ணை', 'ணொ', 'ணோ', 'ணௌ',
            'த்', 'த', 'தா', 'தி', 'தீ', 'து', 'தூ', 'தெ', 'தே', 'தை', 'தொ', 'தோ', 'தௌ',
            'ந்', 'ந', 'நா', 'நி', 'நீ', 'நு', 'நூ', 'நெ', 'நே', 'நை', 'நொ', 'நோ', 'நௌ',
            'ப்', 'ப', 'பா', 'பி', 'பீ', 'பு', 'பூ', 'பெ', 'பே', 'பை', 'பொ', 'போ', 'பௌ',
            'ம்', 'ம', 'மா', 'மி', 'மீ', 'மு', 'மூ', 'மெ', 'மே', 'மை', 'மொ', 'மோ', 'மௌ',
            'ய்', 'ய', 'யா', 'யி', 'யீ', 'யு', 'யூ', 'யெ', 'யே', 'யை', 'யொ', 'யோ', 'யௌ',
            'ர்', 'ர', 'ரா', 'ரி', 'ரீ', 'ரு', 'ரூ', 'ரெ', 'ரே', 'ரை', 'ரொ', 'ரோ', 'ரௌ',
            'ல்', 'ல', 'லா', 'லி', 'லீ', 'லு', 'லூ', 'லெ', 'லே', 'லை', 'லொ', 'லோ', 'லௌ',
            'வ்', 'வ', 'வா', 'வி', 'வீ', 'வு', 'வூ', 'வெ', 'வே', 'வை', 'வொ', 'வோ', 'வௌ',
            'ழ்', 'ழ', 'ழா', 'ழி', 'ழீ', 'ழு', 'ழூ', 'ழெ', 'ழே', 'ழை', 'ழொ', 'ழோ', 'ழௌ',
            'ள்', 'ள', 'ளா', 'ளி', 'ளீ', 'ளு', 'ளூ', 'ளெ', 'ளே', 'ளை', 'ளொ', 'ளோ', 'ளௌ',
            'ற்', 'ற', 'றா', 'றி', 'றீ', 'று', 'றூ', 'றெ', 'றே', 'றை', 'றொ', 'றோ', 'றௌ',
            'ன்', 'ன', 'னா', 'னி', 'னீ', 'னு', 'னூ', 'னெ', 'னே', 'னை',
            'ஶ்', 'ஶ', 'ஶா', 'ஶி', 'ஶீ', 'ஶு', 'ஶூ', 'ஶெ', 'ஶே', 'ஶை', 'ஶொ', 'ஶோ', 'ஶௌ',
            'ஜ்', 'ஜ', 'ஜா', 'ஜி', 'ஜீ', 'ஜு', 'ஜூ', 'ஜெ', 'ஜே', 'ஜை', 'ஜொ', 'ஜோ', 'ஜௌ',
            'ஷ்', 'ஷ', 'ஷா', 'ஷி', 'ஷீ', 'ஷு', 'ஷூ', 'ஷெ', 'ஷே', 'ஷை', 'ஷொ', 'ஷோ', 'ஷௌ',
            'ஸ்', 'ஸ', 'ஸா', 'ஸி', 'ஸீ', 'ஸு', 'ஸூ', 'ஸெ', 'ஸே', 'ஸை', 'ஸொ', 'ஸோ', 'ஸௌ',
            'ஹ்', 'ஹ', 'ஹா', 'ஹி', 'ஹீ', 'ஹு', 'ஹூ', 'ஹெ', 'ஹே', 'ஹை', 'ஹொ', 'ஹோ', 'ஹௌ',
            'க்ஷ்', 'க்ஷ', 'க்ஷா', 'க்ஷ', 'க்ஷீ', 'க்ஷு', 'க்ஷூ', 'க்ஷெ', 'க்ஷே', 'க்ஷை', 'க்ஷொ', 'க்ஷோ', 'க்ஷௌ',
            '்', 'ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', END_TOKEN]
en_vocab = [PADDING_TOKEN, START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            ':', '<', '=', '>', '?', '@',
            '[', '\\', ']', '^', '_', '`',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z', '{', '|', '}', '~', END_TOKEN]

index_to_tamil = {k: v for k, v in enumerate(ta_vocab)}
tamil_to_index = {v: k for k, v in enumerate(ta_vocab)}
index_to_english = {k: v for k, v in enumerate(en_vocab)}
english_to_index = {v: k for k, v in enumerate(en_vocab)}

# === Load and Preprocess Data ===
with open('en-ta/English.txt', 'r', encoding="utf8") as file:
    en_sentences = [line.rstrip('\n').lower() for line in file.readlines()[:200000]]
with open('en-ta/Tamil.txt', 'r', encoding="utf8") as file:
    ta_sentences = [line.rstrip('\n') for line in file.readlines()[:200000]]

def find_invalid_tokens(sentence, vocab):
    return [token for token in set(sentence) if token not in vocab]

def is_valid_token(sentence, vocab):
    return all(token in vocab for token in sentence)

def is_valid_length(sentence, max_length):
    return len(sentence) <= max_length

valid_indices = []
for i, (ta, en) in enumerate(zip(ta_sentences, en_sentences)):
    if (is_valid_length(ta, 250) and is_valid_length(en, 250) and
        is_valid_token(ta, ta_vocab) and is_valid_token(en, en_vocab)):
        valid_indices.append(i)

ta_sentences = [ta_sentences[i] for i in valid_indices]
en_sentences = [en_sentences[i] for i in valid_indices]

# === Dataset & Dataloader ===
def tokenize(sentence):
    return list(sentence)

def tokens_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens]

def add_special_tokens(indices, sos_idx, eos_idx):
    return [sos_idx] + indices + [eos_idx]

class TranslationDataset(Dataset):
    def __init__(self, src_sents, tgt_sents, src_vocab, tgt_vocab):
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_sos = src_vocab['<SOS>']
        self.src_eos = src_vocab['<EOS>']
        self.tgt_sos = tgt_vocab['<SOS>']
        self.tgt_eos = tgt_vocab['<EOS>']

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        src = add_special_tokens(tokens_to_indices(tokenize(self.src_sents[idx]), self.src_vocab), self.src_sos, self.src_eos)
        tgt = add_special_tokens(tokens_to_indices(tokenize(self.tgt_sents[idx]), self.tgt_vocab), self.tgt_sos, self.tgt_eos)
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=english_to_index['<PAD>'])
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tamil_to_index['<PAD>'])
    return src_batch, tgt_batch

dataset = TranslationDataset(en_sentences, ta_sentences, english_to_index, tamil_to_index)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# === Model ===
model = Transformer(
    num_layers=6, d_model=512, dff=2048, dropout=0.1, heads=8,
    src_vocab_size=len(en_vocab), tgt_vocab_size=len(ta_vocab), max_len=252
).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tamil_to_index['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Resume Logic ===
def find_latest_checkpoint(path="output"):
    files = [f for f in os.listdir(path) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if not files:
        return None, 0
    latest = max(files, key=lambda f: int(re.findall(r'\d+', f)[0]))
    epoch = int(re.findall(r'\d+', latest)[0])
    return os.path.join(path, latest), epoch

checkpoint_path, start_epoch = find_latest_checkpoint()
if checkpoint_path:
    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_history = checkpoint.get("loss_history", [])
else:
    print("No checkpoint found. Starting from scratch.")
    start_epoch = 0
    loss_history = []

# === Training Loop ===
num_epochs = 20
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for src_batch, tgt_batch in dataloader:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        src_pad_mask = create_padding_mask(src_batch, english_to_index['<PAD>']).to(device)
        tgt_pad_mask = create_padding_mask(tgt_batch[:, :-1], tamil_to_index['<PAD>']).to(device)
        seq_len = tgt_batch[:, :-1].size(1)
        causal_mask = create_causal_mask(seq_len).to(device)
        combined_mask = combine_masks(tgt_pad_mask, causal_mask).to(device)

        output = model(src_batch, tgt_batch[:, :-1], src_pad_mask, None, combined_mask)
        loss = loss_fn(output.reshape(-1, output.size(-1)), tgt_batch[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history
    }
    torch.save(checkpoint, f"output/checkpoint_epoch_{epoch + 1}.pt")

# === Final Save ===
torch.save(model.state_dict(), "output/final_transformer_model.pt")
print("Final model saved.")

# === Loss Plot ===
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("output/loss_curve.png")
print("Loss curve saved to output/loss_curve.png")