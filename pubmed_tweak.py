from datasets import load_dataset, ReadInstruction

if __name__ == "__main__":
    dataset = load_dataset("pubmed", split=ReadInstruction("train",from_=0, to=10, unit="%", rounding="pct1_dropremainder"))
    print(dataset[0])