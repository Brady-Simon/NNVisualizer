import tkinter as tk
from Sources.Color import Color


class Toolbar:
    """A toolbar that can be drawn through Tkinter.

    The Toolbar allows the user to select negative and positive colors for weights
    and biases for `NNVisualizer`. Colors are stored with `negativeColor` and
    `positiveColor` in RGB format (int, int int) tuples.

    Attributes:
        negativeColor (int, int, int): The RGB value for negative weights/biases.
        positiveColor (int, int, int): The RGB value for positive weights/biases.
    """

    def __init__(self, master, frame, update_func):
        self.master = master
        self.frame = frame
        self.update_func = update_func

        self.negativeColor = (255, 0, 0)
        self.positiveColor = (0, 0, 255)

    def create_widgets(self):
        # Negative Red Text
        self.negativeRedTextField = tk.Text(master=self.master, fg="red", height=1, width=3)
        self.negativeRedTextField.config(highlightbackground="red")
        self.negativeRedTextField.insert(tk.INSERT, "255")
        self.negativeRedTextField.pack(side=tk.LEFT)
        # Negative Green Text
        self.negativeGreenTextField = tk.Text(master=self.master, fg="green", height=1, width=3)
        self.negativeGreenTextField.config(highlightbackground="green")
        self.negativeGreenTextField.insert(tk.INSERT, "0")
        self.negativeGreenTextField.pack(side=tk.LEFT)
        # Negative Blue Text
        self.negativeBlueTextField = tk.Text(master=self.master, fg='blue', height=1, width=3)
        self.negativeBlueTextField.config(highlightbackground="blue")
        self.negativeBlueTextField.insert(tk.INSERT, "0")
        self.negativeBlueTextField.pack(side=tk.LEFT)
        # Negative Label
        self.negativeLabel = tk.Label(master=self.master, background=self.negativeColorHex(),
                                      foreground='white', text='Negative')
        self.negativeLabel.pack(side=tk.LEFT)

        # Positive Blue Text
        self.positiveBlueTextField = tk.Text(master=self.master, fg='blue', height=1, width=3)
        self.positiveBlueTextField.config(highlightbackground="blue")
        self.positiveBlueTextField.insert(tk.INSERT, "255")
        self.positiveBlueTextField.pack(side=tk.RIGHT)
        # Positive Green Text
        self.positiveGreenTextField = tk.Text(master=self.master, fg="green", height=1, width=3)
        self.positiveGreenTextField.config(highlightbackground="green")
        self.positiveGreenTextField.insert(tk.INSERT, "0")
        self.positiveGreenTextField.pack(side=tk.RIGHT)
        # Positive Red Text
        self.positiveRedTextField = tk.Text(master=self.master, fg="red", height=1, width=3)
        self.positiveRedTextField.config(highlightbackground="red")
        self.positiveRedTextField.insert(tk.INSERT, "0")
        self.positiveRedTextField.pack(side=tk.RIGHT)
        # Positive Label
        self.positiveLabel = tk.Label(master=self.master, background=self.positiveColorHex(),
                                      foreground='white', text='Positive')
        self.positiveLabel.pack(side=tk.RIGHT)

        # Update Button
        self.updateButton = tk.Button(master=self.master, text="Update", command=self.update_func)
        self.updateButton.pack(fill=tk.Y, expand=False, anchor=tk.S)

    def updateColors(self):
        """Updates `positiveColor` and `negativeColor` to match the text fields."""
        self.negativeColor = (int(self.negativeRedTextField.get("1.0", tk.END)),
                              int(self.negativeGreenTextField.get("1.0", tk.END)),
                              int(self.negativeBlueTextField.get("1.0", tk.END)))
        self.positiveColor = (int(self.positiveRedTextField.get("1.0", tk.END)),
                              int(self.positiveGreenTextField.get("1.0", tk.END)),
                              int(self.positiveBlueTextField.get("1.0", tk.END)))
        # Update the positive and negative labels
        self.negativeLabel.config(background=self.negativeColorHex())
        self.positiveLabel.config(background=self.positiveColorHex())

        print(f"Negative: {self.negativeColor}")
        print(f"Positive: {self.positiveColor}")

    def positiveColorHex(self) -> str:
        return Color.colorToHex(self.positiveColor)

    def negativeColorHex(self) -> str:
        return Color.colorToHex(self.negativeColor)
