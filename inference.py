import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer import Transformer, create_padding_mask, create_causal_mask, combine_masks
import heapq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)
os.makedirs("output", exist_ok=True)

START_TOKEN = '<SOS>'
PADDING_TOKEN = '<PAD>'
END_TOKEN = '<EOS>'
UNKNOWN_TOKEN = '<UNK>'

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

model = Transformer(
    num_layers=6, d_model=512, dff=2048, dropout=0.1, heads=8,
    src_vocab_size=len(en_vocab), tgt_vocab_size=len(ta_vocab), max_len=252
).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tamil_to_index['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

MODEL_PATH = r"output\final_transformer_model.pt"

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print(f'Model loaded from {MODEL_PATH}')


def translate(sentence, model,
              english_to_index, index_to_tamil,
              max_length=250):
    model.eval()
    tokens = tokenize(sentence.lower())
    indices = tokens_to_indices(tokens, english_to_index)
    indices = add_special_tokens(indices,
                                 english_to_index[START_TOKEN],
                                 english_to_index[END_TOKEN])
    src_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    src_padding_mask = create_padding_mask(src_tensor, pad_token=english_to_index[PADDING_TOKEN]).to(device)
    tgt_indices = [tamil_to_index[START_TOKEN]]
    tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_length):
        tgt_padding_mask = create_padding_mask(tgt_tensor, pad_token=tamil_to_index[PADDING_TOKEN]).to(device)
        causal_mask = create_causal_mask(tgt_tensor.size(1)).to(device)
        combined_mask = combine_masks(tgt_padding_mask, causal_mask)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor,
                           src_padding_mask,
                           tgt_padding_mask,
                           combined_mask)

        next_token_logits = output[0, -1, :]
        _, next_token = torch.max(next_token_logits, dim=-1)
        next_token = next_token.item()
        tgt_indices.append(next_token)
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)

        if next_token == tamil_to_index[END_TOKEN]:
            break

    translated_tokens = [index_to_tamil[idx] for idx in tgt_indices[1:] if idx != tamil_to_index[END_TOKEN]]
    translated_sentence = ''.join(translated_tokens)

    return translated_sentence

def translate_beam_search(sentence, model,
                          english_to_index, index_to_tamil,
                          max_length=250, beam_width=3):

    model.eval()

    # Preprocess the input sentence
    tokens = tokenize(sentence.lower())
    indices = tokens_to_indices(tokens, english_to_index)
    indices = add_special_tokens(indices,
                                 english_to_index[START_TOKEN],
                                 english_to_index[END_TOKEN])
    src_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, src_seq_len]

    # Create source padding mask
    src_padding_mask = create_padding_mask(src_tensor, pad_token=english_to_index[PADDING_TOKEN]).to(device)

    # Initialize the beam with the start token
    beams = [([tamil_to_index[START_TOKEN]], 0.0)]  # List of tuples: (sequence, cumulative log-prob)

    completed_beams = []

    for _ in range(max_length):
        new_beams = []
        for seq, score in beams:
            # If the last token is <EOS>, add the beam to completed_beams
            if seq[-1] == tamil_to_index[END_TOKEN]:
                completed_beams.append((seq, score))
                continue

            # Prepare target tensor
            tgt_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, seq_len]

            # Create target padding mask
            tgt_padding_mask = create_padding_mask(tgt_tensor, pad_token=tamil_to_index[PADDING_TOKEN]).to(device)

            # Create causal mask for target
            causal_mask = create_causal_mask(tgt_tensor.size(1)).to(device)

            # Combine masks
            combined_mask = combine_masks(tgt_padding_mask, causal_mask)

            # Forward pass through the model
            with torch.no_grad():
                output = model(src_tensor, tgt_tensor,
                               src_padding_mask,
                               tgt_padding_mask,
                               combined_mask)  # Shape: [1, seq_len, tgt_vocab_size]

            # Get the logits for the last token
            next_token_logits = output[0, -1, :]  # Shape: [tgt_vocab_size]

            # Compute log probabilities
            log_probs = nn.functional.log_softmax(next_token_logits, dim=-1)  # Shape: [tgt_vocab_size]

            # Get the top `beam_width` tokens and their log probabilities
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            # Expand each beam with each of the top `beam_width` tokens
            for i in range(beam_width):
                next_token = topk_indices[i].item()
                next_log_prob = topk_log_probs[i].item()
                new_seq = seq + [next_token]
                new_score = score + next_log_prob
                new_beams.append((new_seq, new_score))

        # If no new beams are generated, break
        if not new_beams:
            break

        # Keep the top `beam_width` beams based on cumulative score
        beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[1])

        # If all beams are completed, stop early
        if len(completed_beams) >= beam_width:
            break

    # If no completed beams, use the current beams
    if not completed_beams:
        completed_beams = beams

    # Select the beam with the highest score
    best_beam = max(completed_beams, key=lambda x: x[1])
    tgt_indices = best_beam[0]

    # Convert indices to tokens, excluding <SOS> and <EOS>
    translated_tokens = [index_to_tamil[idx] for idx in tgt_indices[1:] if idx != tamil_to_index[END_TOKEN]]
    translated_sentence = ''.join(translated_tokens)

    return translated_sentence

english_sentence = "It's not your fault"
tamil_translation = translate(english_sentence, model,
                                english_to_index, index_to_tamil)
print(f"English: {english_sentence}")
print(f"Tamil: {tamil_translation}")

test_sentences = ["How are you brother?"]

for english_sentence in test_sentences:
    tamil_translation_greedy = translate(english_sentence, model,
                                        english_to_index, index_to_tamil)
    tamil_translation_beam = translate_beam_search(english_sentence, model,
                                                    english_to_index, index_to_tamil,
                                                    beam_width=3)
    print(f"English: {english_sentence}")
    print(f"Tamil (Greedy): {tamil_translation_greedy}")
    print(f"Tamil (Beam Search): {tamil_translation_beam}")