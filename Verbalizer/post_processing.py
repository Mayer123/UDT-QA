from rouge import Rouge 
import argparse
rouge = Rouge(['rouge-1'])
import string
from tqdm import tqdm

def read_text(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    return data

def write_text(data, filename):
    with open(filename, 'w') as fout:
        for line in data:
            fout.write(line+'\n')

def extract_content(text):
    cells = []
    text = text.replace('<H>', '<T>').replace('<R>', '<T>')
    for cell in text.split('<T>'):
        if cell.strip() in ['', '[TABLECONTEXT]', '[title]']:
            continue 
        if cell.strip() not in cells:
            cells.append(cell.strip())
    return ' '.join(cells)

def get_char_seq(sent):
    return ''.join([c for c in sent if c != ' ' and c not in string.punctuation and c.isascii()])

def recover_original_strings(outputs, inputs):
    new_outputs = []
    for i in range(len(inputs)):
        inputs[i] = inputs[i].replace('<H>', '<T>').replace('<R>', '<T>')
        cells = []
        for cell in inputs[i].split('<T>'):
            if cell.strip() in ['', '[TABLECONTEXT]', '[title]']:
                continue 
            if cell.strip() not in cells:
                cells.append(cell.strip().lower())
        sent = outputs[i]
        char_seq = [(c, i) for i, c in enumerate(sent) if c != ' ' and c not in string.punctuation and c.isascii()]
        chars = ''.join([c[0] for c in char_seq])
        indices = [c[1] for c in char_seq]

        replace_pairs = []
        for word in cells:
            if any([c in string.punctuation for c in word]) or not word.isascii():
                m_word = ''.join([c for c in word if c not in string.punctuation and c!= ' ' and c.isascii()])
                if len(m_word) == 0:
                    continue
                occurs = chars.find(m_word)
                if occurs >= 0:
                    target_start = indices[occurs]
                    if occurs+len(m_word) < len(indices):
                        target_end = indices[occurs+len(m_word)]
                    else:
                        target_end = len(sent)
                    end_fail = False                   # Here, some punctuation in the end may also be removed, we bring them back
                    while sent[target_end-1] != word[-1]:
                        target_end -= 1
                        if (sent[target_end-1] != ' ' and sent[target_end-1] not in string.punctuation and sent[target_end-1].isascii()):
                            break
                    if sent[target_end-1] != word[-1]:
                        end_fail = True
                    if end_fail:
                        continue
                    start_fail = False             # Here we try to include punctuations that may appear in the begining
                    while sent[target_start] != word[0]:
                        target_start -= 1
                        if (sent[target_start] not in string.punctuation and sent[target_start] != ' ' and sent[target_start].isascii()) or target_start <= 0:
                            break
                    if sent[target_start] != word[0]:
                        start_fail = True
                    if start_fail:
                        continue
                    target = sent[target_start:target_end]
                    if len(get_char_seq(target)) != len(get_char_seq(word)):
                        continue
                    if target != word and (target, word) not in replace_pairs:
                        if target_end < len(sent) and sent[target_end] != ' ':
                            target += ' '
                            word += ' '
                        if target_start > 0:
                            target = ' ' + target
                            word = ' ' + word
                        replace_pairs.append((target, word))
        for p in replace_pairs:
            sent = sent.replace(p[0], p[1])
        new_outputs.append(sent)
    return new_outputs

def process_one_file(output_filename, input_filename):
    sentences = read_text(output_filename)
    all_inputs = read_text(input_filename)
    selected_sentences = []
    emptytop1 = 0
    emptyin = 0
    for i in tqdm(range(len(sentences))):
        beams = sentences[i].split('|||')
        inputs = extract_content(all_inputs[i])
        beams[-1] = beams[-1].strip()
        if beams[0] == '':
            emptytop1 += 1
        if '' in beams:
            emptyin += 1
        try:
            scores_get = [(rouge.get_scores(s.lower(), inputs.lower()), b) for b, s in enumerate(beams)]
            scores_get.sort(key=lambda x: x[0][0]['rouge-1']['f'], reverse=True)
            sent = beams[scores_get[0][1]]
        except:
            print (inputs)
            sent = ''
            for beam in beams:
                if beam != '':
                    sent = beam 
                    break
        recover_sent = recover_original_strings([sent], [all_inputs[i]])[0]
        selected_sentences.append(recover_sent)
    return selected_sentences

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbalizer_outputs", default=None, type=str, required=False)    
    parser.add_argument("--verbalizer_inputs", default=None, type=str, required=False)   
    args = parser.parse_args()
    selected_sentences = process_one_file(args.verbalizer_outputs, args.verbalizer_inputs)
    write_text(selected_sentences, args.verbalizer_outputs.replace('.txt', 'beam_selection.txt'))