import tkinter as tk
from Sources.Toolbar import Toolbar
from Sources.StateDictionaries import StateDictionaries


class NNVisualizer(tk.Frame):
    """Visualizes the state dictionary of a neural net.

    Live updates during training are supported through extra threads.
    Follow these general steps:
        - Instantiate your ML model.
        - Inject the ML state dictionary into the NNVisualizer
          and update periodically using `update_state_dict`
          and `update_interval`.
        - Begin training your model on a separate thread
        - Show the NNVisualizer using `show()` or `mainloop()`.

    If you only want to display the fully-trained model, then just pass
    in the final model into the initializer and show the visualizer.
    """

    def __init__(self, state_dict: dict, update_state_dict=None, update_interval: int = -1, master=None):
        """Initializes a new Neural Net Visualizer.

        Args:
            state_dict (dict): The state dictionary to visualize.
            update_state_dict: A function that returns an updated version of the state dictionary.
            update_interval (int): How often (in ms) to check for updated state dictionaries.
                Does not update the state dictionary of the frequency is less than 0.
            master: The TK root to use. One will be created if left empty.
        """
        if master is None:
            self.master = tk.Tk()
            self.master.title("Neural Net Visualizer")
            self.master.resizable()
        else:
            self.master = master
        super().__init__(master)

        self.state_dict = state_dict
        self.update_state_dict = update_state_dict
        self.update_interval = update_interval
        self.canvas = tk.Canvas(master=self.master, width=500, height=400, highlightthickness=0)
        self.canvas.bind("<Configure>", self.rebuild)

        self.toolbar = Toolbar(master=self.master, frame=self, update_func=self.updateColors)
        self.negativeColor = (255, 0, 0)  # Red
        self.positiveColor = (0, 0, 255)  # Blue
        self.create_widgets()
        if update_state_dict is not None and update_interval >= 0:
            self.after(update_interval, lambda: self.updateStateDict(new_state_dict=update_state_dict()))

    def rebuild(self, event=None):
        originalHeight = self.height()
        originalWidth = self.width()
        self.canvas.config(width=event.width, height=event.height)
        if originalHeight != self.canvas.winfo_reqheight() or originalWidth != self.canvas.winfo_reqwidth():
            self.canvas.delete(tk.ALL)
            self.drawNN()

    def updateStateDict(self, new_state_dict):
        """Updates the local state dictionary and redraws the screen."""

        # Refresh the state dictionary and redraw the screen
        self.state_dict = new_state_dict
        self.canvas.delete(tk.ALL)
        self.drawNN()
        self.update()

        # Call this function again if applicable and continue checking for updates
        if self.update_state_dict is not None and self.update_interval >= 0:
            self.after(self.update_interval, lambda: self.updateStateDict(self.update_state_dict()))

    def xStart(self, count: int = None) -> int:
        """Returns the starting x position to use."""
        if count is None:
            return 50
        else:
            return self.incrementAmount(count) // 2

    def incrementAmount(self, horizontalCount: int = None) -> int:
        """The amount to increment horizontally across the screen.

        Args:
            horizontalCount (int): The number of layers in the neural net.

        Returns:
            int: The amount to increase the current X position while drawing.
        """
        if horizontalCount is None:
            return 100
        else:
            width = self.width() // (horizontalCount + 1)
            return width

    def radius(self, count: int = None) -> int:
        """Returns the radius to use for the circles.

        Args:
            count (int): The number of elements being placed vertically.
                More circles will mean a smaller radius.
        Returns:
            int: The radius to use for all circles.
        """
        if count is None:
            return 20
        else:
            return max(10, int(self.height() / (5 * count)))

    def height(self) -> int:
        """The height of the view, or a default value of 400."""
        height = self.canvas.winfo_reqheight()
        return height if height != 1 else 400

    def width(self) -> int:
        """The width of the view, or a default value of 500."""
        width = self.canvas.winfo_reqwidth()
        return width if width != 1 else 500

    @staticmethod
    def yPositions(height: int, count: int) -> list:
        """Generates a list of y positions with the given number of separators.

        Args:
            height (int): The total height of the window.
            count (int): The number of positions to have in the generated list.

        Returns:
            int: Y-coordinates to place `count` items centered on the window.
        """
        separators = height / (count + 1)
        return [separators * i for i in range(1, count + 1)]

    def create_widgets(self):
        """Creates all of the important widgets on screen."""
        self.drawNN()
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.toolbar.create_widgets()

    def updateColors(self):
        """Updates the local `positiveColor` and `negativeColor` to match
        the values of the text of the text fields.
        """
        self.toolbar.updateColors()
        self.positiveColor = self.toolbar.positiveColor
        self.negativeColor = self.toolbar.negativeColor
        self.canvas.delete(tk.ALL)
        self.drawNN()

    def drawNN(self):
        """Draws the neural net on the canvas."""
        # Draw the input layer
        inputCount = len(list(self.state_dict.values())[0][0])
        horizontalCount = len(self.state_dict) // 2
        x = self.xStart(count=horizontalCount)
        self.drawInputCircles(x, self.height(), inputCount, radius=self.radius(inputCount))
        x += self.incrementAmount(horizontalCount=horizontalCount)

        # Draw the hidden and output layers
        for index, listItem in enumerate(self.state_dict.values()):
            # Even items are weights and odd items are biases
            if index % 2 == 0:
                # Weights: draw lines (likely two-dimensional)
                yPositions = self.yPositions(self.height(), len(listItem))
                for weights, yPos in zip(listItem, yPositions):
                    self.drawLines(x, yPos, weights)
            else:
                # Biases: draw circles
                self.drawLayer(x, listItem, self.radius(len(listItem)))
                x += self.incrementAmount(horizontalCount=(len(self.state_dict) // 2))

        # Draw the color boxes above the positive-negative color pickers
        # Negative and Positive Colors
        negativeColorHex = self.rgbToHex(self.negativeColor[0], self.negativeColor[1], self.negativeColor[2])
        positiveColorHex = self.rgbToHex(self.positiveColor[0], self.positiveColor[1], self.positiveColor[2])
        # Draw the negative color
        self.canvas.create_rectangle(
            0, self.height() - 20,
            87, self.height(),
            fill=negativeColorHex, width=0.0,
        )
        # Draw the positive color
        self.canvas.create_rectangle(
            self.width() - 87, self.height() - 20,
            self.width(), self.height(),
            fill=positiveColorHex, width=0.0,
        )

        # The negative label and its shadow
        self.canvas.create_text(45, self.height() - 9, text="Negative", fill="black")
        self.canvas.create_text(44, self.height() - 10, text="Negative", fill="white")
        # The positive label and its shadow
        self.canvas.create_text(self.width() - 43, self.height() - 9, text="Positive", fill="black")
        self.canvas.create_text(self.width() - 44, self.height() - 10, text="Positive", fill="white")

    def drawCircle(self, x: int, y: int, r: int, color: str, outline: str = "grey"):
        """Draws a circle onto the canvas.

        Args:
            x (int): The X-coordinate to place the circle.
            y (int): The Y-coordinate to place the circle.
            r (int): The radius of the circle.
            color (str): The color to fill the circle.
            outline (str): The outline of the circle. Defaults to 'grey'.
        """
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=outline)

    def drawLayer(self, xPos: int, biases: list, radius: int = 20):
        for bias, yPos in zip(biases, self.yPositions(self.height(), len(biases))):
            self.drawCircle(xPos, yPos, radius, self.numToColor(bias))

    def drawInputCircles(self, xPos: int, height: int, count: int, radius: int = 20):
        yPositions = self.yPositions(height, count)
        for yPos in yPositions:
            self.drawCircle(xPos, yPos, radius, color="#EEEEEE")

    @staticmethod
    def rgbToHex(r: int, g: int, b: int) -> str:
        """Converts the given RGB values (0-255) into hex format, #RRGGBB.

        Args:
            r (int): The amount of red, between 0 and 255.
            g (int): The amount of green, between 0 and 255.
            b (int): The amount of blue, between 0 and 255.

        Returns:
            str: The string representation of the hex color (#RRGGBB).
        """
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    def numToColor(self, num: float) -> str:
        """Interpolates the number (0.0-1.0) into a color.

        Args:
            num (float): The number (0.0-1.0) to convert into a number.

        Returns:
            str: The hex value of the number converted to a string (#RRGGBB).
        """
        progress = (num + 1) / 2
        red = self.clamp(self.lerp(self.negativeColor[0], self.positiveColor[0], progress))
        green = self.clamp(self.lerp(self.negativeColor[1], self.positiveColor[1], progress))
        blue = self.clamp(self.lerp(self.negativeColor[2], self.positiveColor[2], progress))
        return self.rgbToHex(red, green, blue)

    @staticmethod
    def lerp(initial, final, progress):
        """Lerps between `initial` and `final` based on `progress`."""
        return initial * (1 - progress) + final * progress

    @staticmethod
    def clamp(n, smallest: int = 0, largest: int = 255) -> int:
        """Clamps `n` between `smallest` and `largest`."""
        return int(max(smallest, min(n, largest)))

    def drawLine(self, x1: int, y1: int, x2: int, y2: int, color: str, width: float = 2):
        self.canvas.create_line((x1, y1), (x2, y2), fill=color, width=width)

    def drawLines(self, xPos: int, yPos: int, lineWeights: list):
        """Draws lines between the left and right.

        Args:
            xPos (int): The starting x position to draw lines.
            yPos (int): The starting y position to draw lines.
            lineWeights (int): The weight of each line.
        """
        x = xPos - self.incrementAmount(horizontalCount=(len(self.state_dict) // 2))
        for weight, y in zip(lineWeights, self.yPositions(self.height(), len(lineWeights))):
            lineWidth = self.clamp(abs(weight), smallest=0, largest=1) + 1
            self.drawLine(xPos, yPos, x, y, color=self.numToColor(weight), width=lineWidth)

    def show(self):
        """Opens the visualizer on screen. Blocks the thread until
        the window is closed.

        This is just a shortcut for `self.mainloop()`.
        """
        self.mainloop()


isDefaultDictionary: bool = False
"""bool: Whether or not the default dictionary is being shown on screen."""


def update() -> dict:
    """Toggles the global `isDefaultDictionary` and returns the new dictionary to use."""
    global isDefaultDictionary
    isDefaultDictionary = not isDefaultDictionary
    if isDefaultDictionary:
        return StateDictionaries.default_state_dict()
    else:
        return StateDictionaries.snake_state_dict()


def main():
    visualizer = NNVisualizer(state_dict=StateDictionaries.tictactoe_state_dict())
    visualizer.mainloop()


if __name__ == '__main__':
    main()
