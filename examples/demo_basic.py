from ..core.nlq import NLQ

nlq = NLQ()

text = "Apple is investing $5 billion in artificial intelligence."

result = nlq.analyze(text)

print("Tokens:")
for t in result["tokens"]:
    print(t)

print("\nEntities:")
for e in result["entities"]:
    print(e)

print("\nDependencies:")
for d in result["dependencies"]:
    print(d)
