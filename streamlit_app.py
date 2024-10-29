import streamlit as st
import torch
from torch import nn
import re

# Load the trained model
class NextToken(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size_1, hidden_size_2, activation):
    # Initialize the parent class (nn.Module)
    super().__init__()

    # Create an embedding layer to map each character to a vector
    self.emb = nn.Embedding(vocab_size, emb_dim)

    # hidden layers
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size_1)
    self.lin2 = nn.Linear(hidden_size_1, hidden_size_2)

    # Create a linear layer to map the hidden state to the vocabulary size
    self.lin3 = nn.Linear(hidden_size_2, vocab_size)

    self.activation = activation

  def forward(self, x):
    # Embed the input characters
    x = self.emb(x)

    # Reshape the embedding to a 2D tensor
    # Before the dimension was (batch_size, block_size, emb_dim) and now its (batch_size, block_size * emb_dim)
    x = x.view(x.shape[0], -1)

    x = self.activation(self.lin1(x))
    x = self.activation(self.lin2(x))

    x = self.lin3(x)
    return x


def preprocess_text(text):
    # consider text between start and end of project gutenberg. starts after 2nd "***", ends before 3rd "***" and we should not include "***"
    start = text.find("***", text.find("***") + 1)
    end = text.find("***", start + 1)
    text = text[start+3:end]

    # remove all occurences of '&c'
    text = re.sub('&c', '', text)

    # replace new lines with single new line
    text=re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    # replace multiple spaces with single space
    text = re.sub(' +', ' ', text)

    return text

# function to space-seperate per our vocabulary
def tokenize(text):
    formatted_text = ""

    for i in range(len(text)):
        char = text[i]
        next_char = text[i + 1] if i < len(text) - 1 else None

        if char in "-—":
            formatted_text += " " + char + " "
        elif char in "½’?!,)”;":
            formatted_text += " " + char
        elif char in "‘“(":
            formatted_text += char + " "
        elif char in "_0123456789.:":
            if next_char and (not next_char==" ") and (not text[i - 1]==" "):
                formatted_text += " " + char + " "
            elif next_char and next_char==" " and (not text[i - 1]==" "):
                formatted_text += " " + char
            elif next_char and not next_char==" ":
                formatted_text += char + " "
            else:
                formatted_text += char
        else:
            formatted_text += char

    result = ""
    for i in range(len(formatted_text)):
        if formatted_text[i] == "\n":
            # Check if there's a character before '\n' and if it's not a space
            if i > 0 and formatted_text[i - 1] != " ":
                result += " "
            
            # Add the newline character itself
            result += "\n"
            
            # Check if there's a character after '\n' and if it's not a space
            if i < len(formatted_text) - 1 and formatted_text[i + 1] != " ":
                result += " "
        else:
            result += formatted_text[i]
    return result

def detokenize(text):
    detokenized_text = ""
    length = len(text)
    i = 0
    
    while i < length:
        char = text[i]
        
        # Remove spaces before punctuation marks
        if char in ".?!,;”’)":
            if detokenized_text and detokenized_text[-1] == " ":
                detokenized_text = detokenized_text[:-1]
            detokenized_text += char
            
        # Remove spaces around hyphen and em dash in the middle of words
        elif char == '-':
            if detokenized_text and detokenized_text[-1] == " ":
                detokenized_text = detokenized_text[:-1]
            detokenized_text += char
            if i + 1 < length and text[i + 1] == " ":
                i += 1
            
        elif char == '—':
            if detokenized_text and detokenized_text[-1] != " ":
                detokenized_text += " "
            detokenized_text += char
            if i + 1 < length and text[i + 1] != " ":
                detokenized_text += " "
        
        # Remove unnecessary space after '‘', '“', '('
        elif char in "‘“(":
            detokenized_text += char
            if i + 1 < length and text[i + 1] == " ":
                i += 1  # Skip the space after these characters
        
        # Remove spaces around underscores in words
        elif char == "_":
            if detokenized_text and detokenized_text[-1] == " ":
                detokenized_text = detokenized_text[:-1]
            detokenized_text += char
            if i + 1 < length and text[i + 1] == " ":
                i += 1
        
        # Handle numbers with spaces (e.g., "£ 1 0 0" -> "£ 100")
        elif char.isdigit():
            # Check if previous character was a space and previous non-space was a number
            if detokenized_text and detokenized_text[-1] == " " and detokenized_text[-2].isdigit():
                detokenized_text = detokenized_text[:-1]
            detokenized_text += char

        # Handle newlines with no spaces around them
        elif char == "\n":
            # Remove space before newline
            if detokenized_text and detokenized_text[-1] == " ":
                detokenized_text = detokenized_text[:-1]
            detokenized_text += char
            # Skip any space right after newline
            if i + 1 < length and text[i + 1] == " ":
                i += 1
        
        # Leave '£' and '&' as is
        elif char in "£&":
            detokenized_text += char
        
        # Add all other characters as they are
        else:
            detokenized_text += char
        
        i += 1
    
    return detokenized_text.strip()


# Function to generate sentences
def generate_per_context(model, itow, wtoi, context, num_paragraphs):
    output = " ".join(context)
    # replace all occurence of "#" (if any) with "..." in output
    output = re.sub('#', ' ', output)
    context = [wtoi[ch] for ch in context]
    counter = 0
    while(True):
        x = torch.tensor(context).view(1, -1).to(device) # [[wtio('I'), wtoi('wake'), wtoi('up')...]]
        y_pred = model(x) # gives logits for the next token
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item() # now we sample from those vocab_size logits
        word = itow[ix] # convert index to word
        output += " " + word
        # we don't have # in our input data.
        if word == "\n":
            counter += 1
            if counter == num_paragraphs:
                break
        context = context[1:] + [ix]
    return output


# Streamlit app
st.title("MLP-based Next Word Prediction")
st.write("A simple MLP model based on The Project Gutenberg eBook of The Adventures of Sherlock Holmes, by Arthur Conan Doyle for next word prediction.")

file_path = "./sherlock_holmes.txt"

# Open the file and read the text
try:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    st.write("File loaded successfully!")

    # Now you can use the 'text' variable for further processing in your application
    with st.expander("View file content", expanded=False):
        st.text_area("Content of the file:", text, height=300)
except FileNotFoundError:
    st.write("File not found. Please check the file path.")

text = preprocess_text(text)
text = tokenize(text)

# Create vocabulary
# vocabulary = sorted(list(set(" ".join(text).split())))
# wtoi = {word: i for i, word in enumerate(vocabulary)}
# itow = {i: word for i, word in enumerate(vocabulary)}

# Model configuration
block_size = st.sidebar.selectbox("Context Length", [5, 10, 15])

emb_dim = st.sidebar.selectbox("Embedding Dimensions", [64, 128])

activation_functions = {
    'relu': torch.relu,
    'tanh': torch.tanh
}


activation = st.sidebar.selectbox("Activation Function", ["relu", "tanh"])
# print("======================================")
# print(type(activation))
activation_function = activation_functions[activation]

num_paragraphs = st.sidebar.slider("Number of Paragraphs", 1, 50, 1)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_path = f"question_1_models/next_word_model_{block_size}_{emb_dim}_{activation}.pt"

# Load pre-trained model from a specific path
# Load pre-trained model from a specific path
model = NextToken(block_size, 8459, emb_dim, 2048, 2048, activation_function).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
model = model.to(device)

vocabulary = sorted(list(set(re.split(r'[ \t\r\f\v]+', text))))
wtoi = {ch:i for i, ch in enumerate(vocabulary)}
itow = {i:ch for i, ch in enumerate(vocabulary)}

# add '#' to the wtoi and itow mappings to represent empty
index = len(wtoi)
wtoi['#'] = index
itow[index] = '#'

print(len(itow))

tSNE_path = f"question_1_tsne/tsne_embeddings_{block_size}_{emb_dim}_{activation}.html"

# Generate text
st.subheader("Generate Text")
user_input = st.text_input("Enter some context:").strip()
if st.button("Generate"):
    user_input = tokenize(user_input)

    user_context = user_input.split(" ")
    if len(user_context) >= block_size:
        user_context = user_context[-block_size:]
    else:
        user_context = ["#"] * (block_size - len(user_context)) + user_context
    output = generate_per_context(model, itow, wtoi, user_context, num_paragraphs)
    output = detokenize(output)
    output = re.sub('\n', '\n\n', output)

    st.write(output)

# st.subheader("tSNE Visualization of Word Embeddings")
# if st.button("Untrained"):
#     with open('task_1_tnse/tsne_visualization_untrained.html', 'r') as f:
#         html_content = f.read()
    
#     # Render the HTML content interactively
#     st.components.v1.html(html_content, height=600, scrolling=True)

# if st.button("Trained"):
#     with open(tSNE_path, 'r') as f:
#         html_content = f.read()
    
#     # Render the HTML content interactively
#     st.components.v1.html(html_content, height=600, scrolling=True)