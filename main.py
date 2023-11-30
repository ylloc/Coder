import traceback
from contextlib import redirect_stdout
from io import StringIO

from sentencepiece import SentencePieceProcessor

from examples.llama import Transformer
from tinygrad import Tensor, nn
from extra.models.llama import convert_from_huggingface
from tinygrad.helpers import Timing, fetch, colored, getenv
import os, sys


def create_model_cache(output_file, model):
    with Timing("download weights: "):
        part1 = nn.state.torch_load(fetch(
            "https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/resolve/main/pytorch_model-00001-of-00002.bin?download=true"))
        part2 = nn.state.torch_load(fetch(
            "https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/resolve/main/pytorch_model-00002-of-00002.bin?download=true"))

    with Timing("weights -> model: "):
        nn.state.load_state_dict(model, convert_from_huggingface(part1, model, 32, 8), strict=False)
        nn.state.load_state_dict(model, convert_from_huggingface(part2, model, 32, 8), strict=False)

    with Timing("saving float16 cache: "):
        nn.state.safe_save(nn.state.get_state_dict(model), output_file)

    print("cache created, rerun to use")
    exit(0)


def create_fixed_tokenizer(output_file):
    print("creating fixed tokenizer")
    import extra.junk.sentencepiece_model_pb2 as spb2
    mp = spb2.ModelProto()
    mp.ParseFromString(fetch(
        "https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/resolve/main/tokenizer.model?download=true").read_bytes())
    mp.pieces.append(spb2.ModelProto.SentencePiece(piece="<|im_end|>", score=0))
    mp.pieces.append(spb2.ModelProto.SentencePiece(piece="<|im_start|>", score=0))
    with open(output_file, "wb") as f:
        f.write(mp.SerializeToString())


if __name__ == "__main__":
    tiny = {
        "dim": 2048,
        "n_layers": 22,
        "n_heads": 32,
        "n_kv_heads": 4,
        "norm_eps": 1e-05,
        "vocab_size": 32000,
        "hidden_dim": 5632
    }

    Tensor.no_grad = True
    with Timing("create model"):
        model = Transformer(**tiny)
    cached_model = "/tmp/cached_openhermes.safetensors"
    if not os.path.isfile(cached_model): create_model_cache(cached_model, model)
    with Timing("loading float16 cache: "):
        nn.state.load_state_dict(model, nn.state.safe_load(cached_model))

    if not os.path.isfile("/tmp/tokenizer.model"): create_fixed_tokenizer("/tmp/tokenizer.model")
    spp = SentencePieceProcessor(model_file="/tmp/tokenizer.model")
    print("Model loaded!\n" + 100 * "-")

    END_T, START_T = 32000, 32001


    def encode_prompt(k, v):
        return [START_T] + spp.encode(f"{k}\n{v}") + [END_T] + spp.encode("\n")


    def start_prompt(k):
        return [START_T] + spp.encode(f"{k}\n")


    def output(outputted, toks, color):
        # print(toks)
        cur = spp.decode(toks)[len(outputted):]
        sys.stdout.write(colored(cur, color))
        sys.stdout.flush()
        outputted += cur
        return outputted


    toks = [spp.bos_id()] + encode_prompt("system",
                                          "You are Quentin. Quentin is a useful assistant who writes Python code to answer questions. He keeps the code as short as possible and doesn't read from user input")
    PROMPT = getenv("PROMPT", 1)
    temperature = getenv("TEMP", 0.7)
    start_pos = 0
    outputted = output("", toks, "green")
    turn = True
    while 1:
        if PROMPT:
            toks += encode_prompt("user", input("Q: ")) + start_prompt("assistant")
        else:
            toks += start_prompt("user" if turn else "assistant")
            turn = not turn
        old_output_len = len(outputted)
        while 1:
            tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).multinomial().item()
            start_pos = len(toks)
            toks.append(tok)
            outputted = output(outputted, toks, "blue" if not turn else "cyan")
            if tok == END_T: break
            if tok == spp.eos_id(): break
            new_output = outputted[old_output_len:]

            if new_output.endswith("```") and '```python\n' in new_output:
                python_code = new_output.split('```python\n')[1].split("```")[0]
                # AI safety. Warning to user. Do not press y if the AI is trying to do unsafe things.
                if input(colored(f" <-- PYTHON DETECTED, RUN IT? ", "red")).lower() == 'y':
                    my_stdout = StringIO()
                    try:
                        with redirect_stdout(my_stdout):
                            exec(python_code)
                        result = my_stdout.getvalue()
                    except Exception as e:
                        result = ''.join(traceback.format_exception_only(e))
                    toks += spp.encode(f"\nOutput:\n```\n{result}```")
                    outputted = output(outputted, toks, "yellow")
                    old_output_len = len(outputted)
        print("")
