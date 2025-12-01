def parse_dependencies(doc):
    deps = []
    for token in doc:
        deps.append({
            "token": token.text,
            "dep": token.dep_,
            "head": token.head.text
        })
    return deps
