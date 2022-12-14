# HHShape
a tool for shape component extraction in sketch

## Requirments
- tkinter
- numpy
- matplotlib
- opencv-python
- bezier
- bresenham
- shapely

## How to use
Here we take image `panda.png` as example to demonstrate how to use our tool.
You can start by running the following command:
```bash
python decompose.py panda.png
```
After loading image, which may take few minutes, you will see the following screen:  

![](demo.png)

There are red arrows attched to each stroke. What do they do?  

We define region as the place surrounded by strokes.  
It worth noting that each stroke has two sides belonging to two regions A, B.  

The arrow represent the orientation of the stroke, i.e., whether or not the region A occludes region B by this stroke. if A occludes B, then the arrow of this stroke will point from A to B, and vice versa.  

First, our algorithm will automatically find out a initial configuration of arrows, based on local convexity of shapes.  
Next, Monte Carlo Tree Search will be used to adjust the arrows for optimizing the objective function which measures how good the decomposition is.   
After that, you can interactively modify the result if needed.  

You can do:
- Press <kbd>Enter</kbd> to decompose shape components of this image.
- Press <kbd>Esc</kbd> to go back to initial screen.
- Click an arrow to change its direction and then the result of decomposition.
- Draw a approximate contour to select a shape component (with automatically local arrows adjustment). Press <kbd>a</kbd> to finish drawing.

![](demo.gif)


## License

MIT License