# Artificial-Intelligence
## Getting Started
We built the AI which contains the car to train to drive itself and avoid abstacles on which we draw some roads and blocks for car to navigate around that.
So we use two files ai.py whcih is for slef drive and map.py which for complete environment.
## Self Driving Car Step 1
We are initializing the map with map.py file.
### 1. Import libraries
````
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
```` 
### 2. Import Kivy Packages
````
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
````
### 3. Importing the Dqn object from our AI in ai.py
`from ai import Dqn`

Dqn itslef is AI. It stands for Deep Q networks. We implement Dqn and import in this code by ai.py 

### 4. Adding this line if we don't want the right click to put a red point
`Config.set('input', 'mouse', 'mouse,multitouch_on_demand')`

### 5. Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
`last_x = 0
last_y = 0
n_points = 0
length = 0`

Here `last_x = 0` and `last_y = 0` are the last cordinates of sand on map. When car reaches to sand path car slow downs its drving speed.
### 6. Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
`brain = Dqn(5,3,0.9)
action2rotation = [0,20,-20]
last_reward = 0
scores = []`

We get AI which is called as brain. so it contains neurol network. `brain = Dqn(5,3,0.9)` Here brain is object and Dqn is the class. 
**5,3,0.9**  are the inputs of the class. 
- 5 means **states** which are encoded vector of 5 dimentional.Descrides what happening on the map.
- 3 means actions i.e go left, go right, go straight.
- 0.9 is the gamma parameter in deep Q learning algorithm.

`action2rotation = [0,20,-20]` **Actions** are encoded by 3 numbers. When the index is zero the action goes to 0 value that is car moves straight. When the index is 1 then car moves 20 degrees towards right and when the index is 2 then car moves -20 degrees left. 
`last_reward = 0` When the car goes at sand the **reward** is bad and when it won't goes the reward is good. `scores = []` it contains rewards only at the time of curves.
### 7. Initializing the map
```
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False
```
    
Sand is the array of pixels if it is 1 then there is sand.If it is 0 then there is no sand.Started there won't be sand so it is initialized as `sand = np.zeros((longueur,largeur))` . ` goal_x = 20` goal means we trained the car to reach the destination. Here we are using 20 inorder to not to toch the walls of the map. 
### 8. Initializing the last distance
`last_distance = 0` it gives the current distance to last distance of the car on the road. 
## Self Driving Car Step 2
Now we create car and the game by AI
### 9. Creating the car class
```
class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
``` 
We are initializing the car. **Angle** it says about the angle between X and Y axis i.e, direction of the car. **Rotation** it says about the [last rotation](#6-getting-our-ai-which-we-call-brain-and-that-contains-our-neural-network-that-represents-our-q-function) .
**Sensors** there are 3 if car sense any thing in front then sensor1 is activated and for rght sensor2 and for left sensor3. The **signals** from the sensors are considered as Signal,2,3 respectively.
Then we go for function move ` def move(self, rotation)` the function is updated by considering the last values of the velocity and rotation. Also when the car reaches to the corners of the wall then we consider the signal as 1.

### 10. Creating the game class
```
class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
        last_distance = distance
 ```
Game is to avoid the obstacles and is designed by AI .` action = brain.update(last_reward, last_signal)` It is the action of the car and this action is  output from neurol network.Because NN is the heart of the AI. **[Brain](#6-getting-our-ai-which-we-call-brain-and-that-contains-our-neural-network-that-represents-our-q-function)** it is the object and having the method called update.**Orientation** is the of the input and **- orientation** stablizes the car for exploration i.e it can move right or left. When the car is on sand it slow downs and gets negative reward. We calculate the mean of rewards `scores.append(brain.score())`. We update `goal_x` and `goal_y` to calculate the distance. 
 
### 11. Adding the painting tools
```
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y
 ```
 We create the paintings on the raods.
### 12. Adding the API Buttons (clear, save and load)
```
class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()
```
Here API buttons. Save button saves the memory of the car and if we go for load then the car is undergo for trained.
### 13. Running the whole thing
```
if __name__ == '__main__':
    CarApp().run()
```

Now we implement the DQN class in AI and then create a method for that DQN class inorder to move the car in proper manner rather than randomly as we saw in above.

## Self Driving Car - Step 3
### Importing the libraries
```
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
```
**Torch** we are implementing NN by pytorch specially for AI. torch.nn is so important for the output from the softmax function.
**OS** used to load the model and also reuse the model
**random** for random samples for different vectors.
## Self Driving Car - Step 4
### Creating the architecture of the Neural Network
```
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
 ```
 We are making inherient class. `Self` is used to specify the varible. Input neurons are `input_size`. output neuron is `nb_action`. `Super` function is used to use more effectively the inherient things. Now we create full connection between input layeras and hidden layers by `self.fc1` and second full coonection from hidden layer to output layer given as `self.fc2`.
 ## Self Driving Car - Step 5  
 ```
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
  ```
 `forward` function is used to activate the hidden layers. So we are using relu i.e is the rectifier activation function from the F of the torch. Then we get activated hidden neuron as `x`. output neurons of the `q_values` are consider from the full concection of `self.f2`
 ## Self Driving Car - Step 6
 ### Implementing Experience Replay
```
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
```
We create next `event` by considering last memeory i.e by `capacity`. so the memory is initialized as `self.memory=[]`. Now `push` function is made inorder to append an event in existing memeory. Event is based on *last state,new state,last action, last reward*. Now we need to sample the vector randomly so we create `sample` function. So in order to get sample we consider memory and batch_size. Here from that `batch_size` we consider randomly the samples. Also zip * reshapes the array.

Here we considering `Zip` fucntion then from random library we considering sample function.
In each row action and rewards at same time are concantination.Then convert them into torch values.lamba value is applied to old memeory values in order to get new torch values.
 ## Self Driving Car - Step 7
 #### Implementing Deep Q Learning
```
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
 ```
We are using `Dqn`.We need optimizer for performing the stocasthic gradient decent.`input_size` is the *states* and `nb_actions` is the *actions* taken by car. `gamma` parameter is the parameter of the equation. `reard_window` is the sliding window which have the mean of the reward of previous.`self.model` creates the neural network.**Optimizer** we choose `adam` for best performance.`lr` is used to load the things slowly if it is fast then AI won't learn properly i.e self driving car won't give punishments. Here `last_state` is used for creating tensor class. Becuase for vectorization of *orientation and - orientation* we need `torch.tensor`i.e it contains gradient..Also we need fake dimension so we using `unsqueeze`. [action](#6-getting-our-ai-which-we-call-brain-and-that-contains-our-neural-network-that-represents-our-q-function) is initilized with zero.
```
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial()
        return action.data[0,0]
 ```
Here we get action from the outpout of the NN.So ouput of NN depends directly depends on input.we get q values for all states. Then we use `softmax` to play best action value.we get probabilites so that we can use the all of them by there values. Here we need to change torch.Tensor to variable. By having the more temperature T value we have best appearence of car and best q value. Then for random actions we go for multinomial for the `probs`.
```
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
 ```
According to **Markove decision process** function we need transitions,So we are good for sampling i.e,batch transitions.So in order to use best action we go for `gather` and we make `batch_action`in simple vector so we are using `unsqueeze()`.Then for killing fake dimension we are using `squeeze()`.Now input of the NN is` batch_next_state` so for that output we are considering with *max q value*.
**Target**  is got by adding the next output and reward. for that we need to caluclate loss.so loss is calculated as TD loss.Inorder to improve the loss we are using backpropagation for retain_variables as True.
```    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
 ```
Now making the connection between AI and [Game](#10-creating-the-game-class). So we need to update `new_state`. So new_signal is simple list so we need to covert to torch.Tensor for deep NN. Now update the memory by adding the new state. ie, by using the `push` function.
For new state we need to play action by taking `self.select_action`. If the action is good then we get reward so need to update the reward by `reward_window`.
 ```
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
 ```
Here all the rewards are considered for making the mean.In orddr to avoid denominator zero we are adding with 1.
 ```
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
 ```
Here the last weights and optimizers are saved.so to save we are using `torch.save`.so we need to keys and must be saved them in dictionary. First key is `self.model` and second key is `Optimizer`.So it is saved in the `last_brain.pth`.
``` 
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
```
Now after saving we can load them in self model.`OS.path` is leads working drive folder.`load_state_dict` method is used to load `state_dict` and `optimizer`.
