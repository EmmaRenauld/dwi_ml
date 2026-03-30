from sphinx.application import Sphinx

"""
This is useful to get a map of the website.

USAGE:
    >> sphinx-build -b dummy docs/ _build
    >> python docs/print_toctree.py
"""

app = Sphinx(
    srcdir="docs",
    confdir="docs",
    outdir="_build",
    doctreedir="_build/.doctrees",
    buildername="dummy",
)

app.build()
env = app.env


def print_sections(source, indent="", is_last=False):
    # 1. Prepare connector.
    connector = "└── " if is_last else "├── "

    # 2. Captions?
    doctree = env.get_doctree(source)
    txt = str(doctree)
    if 'toctree caption=' in txt:
        tmp = txt.split('toctree caption="')
        for section in range(1, len(tmp)):
            info = tmp[section].split('" entries="[')
            caption = info[0]
            caption = "** " + caption + " **"
            print(indent + connector + caption)

            entries = info[1].split("]")[0]
            entries = entries.split(',')
            for entrie in entries:
                if 'None' in entrie:
                    continue
                else:
                    entrie = entrie.strip(')').strip().strip("'")
                    print_tree(entrie, "    ")


def print_tree(filename, indent="", is_last=True):
    connector = "└── " if is_last else "├── "

    # Get the title if it exists, else will print the filename
    title = env.titles.get(filename)
    if title is not None:
        display_name = title.astext()
    else:
        display_name = filename

    # Print the current file
    print(indent + connector + display_name)

    # Now print its children
    children = env.toctree_includes.get(filename, [])
    for i, child in enumerate(children):
        last = i == len(children) - 1
        new_prefix = indent + ("    " if is_last else "│   ")
        print_tree(child, new_prefix, last)

print("Site structure:\n")
print_sections("index")
