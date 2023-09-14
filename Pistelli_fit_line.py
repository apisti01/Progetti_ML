# Linear solver
def mylinfit(x,y):

    # a and b for the Mean square error
    b = (sum(x*x)*sum(y)-sum(x)*sum(x*y))/(sum(x*x)*len(x)-sum(x)*sum(x))
    a = (sum(x*y)*len(x)- sum(x)*sum(y))/(sum(x*x)*len(x)-sum(x)*sum(x))

    return a , b

# Main
#import correct libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

x = np.random.uniform(-2,5,10) # 10 random numbers between -2 and 5
y = np.random.uniform(0,3,10) # 10 random numbers between 0 and 3

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlim(-2,5)
ax.set_ylim(0,3)

#plot the points
a,b = mylinfit(x,y)
plt.plot(x,y,'kx')
xp = np.arange(-2,5,0.1)
plt.plot(xp,a*xp+b,'r-')
print(f"My fit: a={b} and b={b}")

#define what to do on mouse click
def onclick(event):
    if event.button is MouseButton.LEFT:
        left_mouse_click(event)
    elif event.button is MouseButton.RIGHT:
        right_mouse_click(event)

#define what to do on left mouse click
def left_mouse_click(event):
    global x,y
    global ax
    ix, iy = event.xdata, event.ydata # get the x and y pixel coords
    #append the coords to the list
    x = np.append(x,ix)
    y = np.append(y,iy)

    #plot the points
    ax.plot(x,y,'kx')
    print(x)
    plt.draw()

#define what to do on right mouse click
def right_mouse_click(event):
    a,b = mylinfit(x,y) #call the MSE function

    #replotting and drawing
    xp = np.arange(-2,5,0.1)
    plt.plot(xp,a*xp+b,'r-') #plot the line
    print(f"My fit: a={b} and b={b}")
    plt.draw() #redraw


#event loop for mouse clicks
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
