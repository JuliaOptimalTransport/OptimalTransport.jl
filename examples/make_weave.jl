using Weave
filename = normpath("examples.jmd")
weave(filename, out_path = :pwd)
