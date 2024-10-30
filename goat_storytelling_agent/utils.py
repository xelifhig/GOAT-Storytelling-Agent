def split_into_words_w_newline(text):
    lines = text.split('\n')
    split_text = [line.split(None) for line in lines if line]
    return split_text


def remove_last_n_words(text, n):
    split_text = split_into_words_w_newline(text)
    i = 1
    lines_to_slice = 0
    while True:
        line = split_text[-i]
        if line:
            n_words = len(line)
            if n_words < n:
                n -= n_words
                lines_to_slice += 1
            else:
                split_text[-i] = line[:-n]
                break
        i += 1
        if i > len(split_text):
            break
    split_text = split_text[:-lines_to_slice]
    text = "\n".join([" ".join(line) for line in split_text])
    return text.strip()


def keep_last_n_words(text, n):
    if not text or n <= 0:
        return ""
        
    split_text = split_into_words_w_newline(text)
    if not split_text:
        return ""
        
    word_count = 0
    lines_to_keep = []
    
    # Work backwards through the lines
    for line in reversed(split_text):
        if line:  # if line is not empty
            words_in_line = len(line)
            if word_count + words_in_line <= n:
                # Keep the whole line
                lines_to_keep.insert(0, line)
                word_count += words_in_line
            else:
                # Keep partial line
                words_needed = n - word_count
                lines_to_keep.insert(0, line[-words_needed:])
                break
                
        else:  # empty line
            lines_to_keep.insert(0, line)
            
        if word_count >= n:
            break
            
    # Reconstruct the text
    text = "\n".join([" ".join(line) for line in lines_to_keep])
    return text.strip()
