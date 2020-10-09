import tkinter as tk
import random


class NNVisualizer(tk.Frame):

    def __init__(self, state_dict: dict, master=None):
        super().__init__(master)
        self.master = master
        self.state_dict = state_dict
        self.pack()
        self.create_widgets()

    @staticmethod
    def xStart() -> int:
        """Returns the starting x position to use."""
        return 150

    @staticmethod
    def incrementAmount() -> int:
        return 100

    @staticmethod
    def radius() -> int:
        """Returns the radius to use for the circles."""
        return 20

    @staticmethod
    def height() -> int:
        return 400

    @staticmethod
    def width() -> int:
        return 500

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
        self.canvas = tk.Canvas(master=self.master, width=self.width(), height=self.height())

        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        # self.drawLines(xPos=self.xStart() + 100, yPos=200, lineWeights=[0.0, 0.25, 0.5, 0.75, 1.0])
        # self.drawLayer(50, [1.0, 0.8, 0.6, 0.2, 0.0])
        self.drawNN()
        self.canvas.pack(fill=tk.BOTH, expand=1)

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def drawNN(self):
        x = self.xStart()

        # Draw the input layer
        inputCount = len(list(self.state_dict.values())[0])
        print(list(self.state_dict.values())[0])
        self.drawInputCircles(self.height(), inputCount, radius=self.radius())

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
                self.drawLayer(x, listItem, self.radius())
                x += 100

    def say_hi(self):
        print("hi there, everyone!")

    def drawCircle(self, x: int, y: int, r: int, color: str):
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=color)

    def drawLayer(self, xPos: int, biases: list, radius: int = 20):
        for bias, yPos in zip(biases, self.yPositions(self.height(), len(biases))):
            self.drawCircle(xPos, yPos, radius, self.numToColor(bias))

    def drawInputCircles(self, height: int, count: int, radius: int = 20):
        yPositions = self.yPositions(height, count)
        xPos = self.xStart() - self.incrementAmount()
        for yPos in yPositions:
            self.drawCircle(xPos, yPos, radius, color="grey")


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
        red = self.clamp(int(255 * (1 - num)))  # Low value should be red
        green = 0
        blue = self.clamp(int(255 * num))  # High num should be blue
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
        x = xPos - self.incrementAmount()
        for weight, y in zip(lineWeights, self.yPositions(self.height(), len(lineWeights))):
            self.drawLine(xPos, yPos, x, y, color=self.numToColor(weight))


default_state_dict = {'0.weight': [[0.2576, -0.2207, -0.0969, 0.2347],
                                   [-0.4707, 0.2999, -0.1029, 0.2544],
                                   [0.0695, -0.0612, 0.1387, 0.0247],
                                   [0.1826, -0.1949, -0.0365, -0.0450]],
                      '0.bias': [0.0725, -0.0020, 0.4371, 0.1556],
                      '2.weight': [[-0.1862, -0.3020, -0.0838, -0.2157],
                                   [-0.1602, 0.0239, 0.2981, 0.2718],
                                   [-0.4888, 0.3100, 0.1397, 0.4743],
                                   [0.3300, -0.4556, -0.4754, -0.2412]],
                      '2.bias': [0.4391, -0.0833, 0.2140, -0.2324]}

new_state_dict = {'0.weight': [[1.1293, -0.8090, -0.5287, 1.5705],
                               [-1.7147, -0.0526, 2.0345, 0.5660],
                               [0.6976, 1.7807, -0.2015, -0.9580],
                               [0.1019, -0.2117, -0.0835, -0.0862]],
                  '0.bias': [0.3508, 0.1918, 0.4731, 0.1465],
                  '2.weight': [[0.7126, -1.9938, 0.1523, -0.2113],
                               [-1.3194, -0.2110, 1.3227, 0.2657],
                               [-0.9676, 1.8449, -0.4638, 0.4726],
                               [1.3500, 0.1278, -1.7198, -0.2386]],
                  '2.bias': [0.1505, -0.5000, -0.1721, -0.3832]}


def main():
    root = tk.Tk()
    root.title("NN Visualizer")

    visualizer = NNVisualizer(master=root, state_dict=new_state_dict)
    visualizer.mainloop()


if __name__ == '__main__':
    main()
