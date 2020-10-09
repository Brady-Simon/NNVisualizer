import tkinter as tk
from Sources.StateDictionaries import StateDictionaries


class NNVisualizer(tk.Frame):

    def __init__(self, state_dict: dict, master=None):
        super().__init__(master)
        self.master = master
        self.state_dict = state_dict
        self.canvas = tk.Canvas(master=self.master, width=500, height=400, highlightthickness=0)
        self.canvas.bind("<Configure>", self.rebuild)
        self.pack()
        self.create_widgets()

    def rebuild(self, event):
        print(event)
        pass
        originalHeight = self.height()
        originalWidth = self.width()
        self.config(width=event.width, height=event.height)
        self.canvas.config(width=event.width, height=event.height)
        # self.update()
        # self.canvas.update()
        if (originalHeight != self.canvas.winfo_reqheight() or originalWidth != self.canvas.winfo_reqwidth()):
            self.canvas.delete("all")
            self.drawNN()

    @staticmethod
    def xStart() -> int:
        """Returns the starting x position to use."""
        return 50

    def incrementAmount(self, horizontalCount: int = None) -> int:
        if horizontalCount is None:
            return 100
        else:
            width = self.width() / (horizontalCount + 1)
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
        height = self.canvas.winfo_reqheight()
        print(f"Height: {height}")
        return height if height != 1 else 400

    def width(self) -> int:
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
        # self.hi_there = tk.Button(self)
        # self.hi_there["text"] = "Hello World\n(click me)"
        # self.hi_there["command"] = self.say_hi
        # self.hi_there.pack(side="top")

        self.drawNN()
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def drawNN(self):
        self.canvas.update()
        x = self.xStart()

        # Draw the input layer
        inputCount = len(list(self.state_dict.values())[0][0])
        print(list(self.state_dict.values())[0][0])
        self.drawInputCircles(x, self.height(), inputCount, radius=self.radius(inputCount))
        x += self.incrementAmount(horizontalCount=(len(self.state_dict) // 2))

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

    def say_hi(self):
        print("hi there, everyone!")

    def drawCircle(self, x: int, y: int, r: int, color: str):
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=color)

    def drawLayer(self, xPos: int, biases: list, radius: int = 20):
        for bias, yPos in zip(biases, self.yPositions(self.height(), len(biases))):
            self.drawCircle(xPos, yPos, radius, self.numToColor(bias))

    def drawInputCircles(self, xPos: int, height: int, count: int, radius: int = 20):
        yPositions = self.yPositions(height, count)
        for yPos in yPositions:
            self.drawCircle(xPos, yPos, radius, color="#AAAAAA")

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
        red = self.clamp(int(255 * (1 - ((num + 1) / 2))))  # Low value should be red
        green = 0
        blue = self.clamp(int(255 * ((num + 1) / 2)))  # High num should be blue
        return self.rgbToHex(red, green, blue)

    @staticmethod
    def clamp(n: int, smallest: int = 0, largest: int = 255):
        """Clamps `n` between `smallest` and `largest`."""
        return max(smallest, min(n, largest))

    def drawLine(self, x1: int, y1: int, x2: int, y2: int, color: str):
        self.canvas.create_line((x1, y1), (x2, y2), fill=color, width=2)

    def drawLines(self, xPos: int, yPos: int, lineWeights: list):
        """Draws lines between the left and right.

        Args:
            xPos (int): The starting x position to draw lines.
            yPos (int): The starting y position to draw lines.
            lineWeights (int): The weight of each line.
        """
        x = xPos - self.incrementAmount(horizontalCount=(len(self.state_dict) // 2))
        for weight, y in zip(lineWeights, self.yPositions(self.height(), len(lineWeights))):
            self.drawLine(xPos, yPos, x, y, color=self.numToColor(weight))


def main():
    root = tk.Tk()
    root.title("NN Visualizer")
    root.resizable()

    visualizer = NNVisualizer(master=root, state_dict=StateDictionaries.tictactoe_state_dict())
    visualizer.mainloop()


if __name__ == '__main__':
    main()
