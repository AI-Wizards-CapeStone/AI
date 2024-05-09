import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb, amsmath}'

label = r"$$\textcircled { \times }$$"
fig = plt.figure()
text = fig.text(
x=0.5,  # x-coordinate to place the text
y=0.5,  # y-coordinate to place the text
s=label,
horizontalalignment="center",
verticalalignment="center",
fontsize=16,
)

plt.show()
