'''
*
* Team Id: 		eYRCPlus-CS#1046
* Author List: 		GOUDU VARA PRASAD,SHUBHAM KUMAR,SAURAV JYOTI SARMA,
                        TIPPANA SAHACHAR REDDY
* Filename: 		Final_code for e-yantra plus 2015
* Theme: 		COURIER SERVICE
* Functions: 		stop(),select(int),turn(int),direction(list),path_following(),
                        filter(char,int,list),find_packages(img),grid_to_arrays(img),
                        pop(dict),djkstras(dict,int),lins2graph(list[][],list[][]),
                        call(int,list[]),get_puj(),
* Global Variables:	pickup,deliver,d_nodes,path
*
'''

import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import package_client
package_client.package_server_ip='192.168.10.14'

global pickup
pickup=[]
global deliver
deliver=[]
global d_nodes
d_nodes=[]
global path
path=[]

'''

* Function Name:	stop
* Input:		void-> does not take any input
* Output:		resets all the PI GPIO pins PWM to 0 duty cycle
* Logic:		passes 0 to the PWMR.start(),PWMR1.start(),
                        PWML.start(),PWML1.start()
* Example Call:	        stop()
*
'''
def stop():
    PWMR.start(0)
    PWMR1.start(0)
    PWML.start(0)
    PWML1.start(0)

'''

* Function Name:	select
* Input:		x -> integer
* Output:		drifts the robot by controlling the direction of
                        rotation of the motors
* Logic:		within the if conditins the dutycycle of respective
                        motors are changed using ChangeDutyCycle() function
* Example Call:	        select(1)
*
'''
def select(x):
    # x: takes values from 1 to 4 and sets the duty cycle of the left and right motors
    if x==1:
        stop()
        PWMR.ChangeDutyCycle(100)
        PWML.ChangeDutyCycle(100)

    elif x==2:
        stop()
        PWMR1.ChangeDutyCycle(100)
        PWML1.ChangeDutyCycle(100)

    elif x==3:
        stop()
        PWMR1.ChangeDutyCycle(0)
        PWML.ChangeDutyCycle(100)

    elif x==4:
        stop()
        PWMR.ChangeDutyCycle(100)
        PWML1.ChangeDutyCycle(0)

    elif x==0:
        stop()
'''

* Function Name:	turn
* Input:		x -> integer
* Output:		turns the robot by controlling the direction of rotation
                        of the motors
* Logic:		within the if conditins the dutycycle of respective motors
                        are changed using ChangeDutyCycle() function
* Example Call:	        turn(1)
*
'''
def turn(x):
    # x: takes values from 1 to 4 and sets the duty cycle of the left and right motors
    if x==1:
        stop()
        PWMR.ChangeDutyCycle(100)
        PWML.ChangeDutyCycle(100)
        time.sleep(0.9)
        stop()
    if x==2:
        PWMR1.ChangeDutyCycle(100)
        PWML1.ChangeDutyCycle(100)
        time.sleep(0.8)
    if x==3:
        stop()
        PWMR1.ChangeDutyCycle(100)
        PWML.ChangeDutyCycle(100)
        time.sleep(0.8)
        stop()
    if x==4:
        stop()
        select(1)
        time.sleep(0.2)
        stop()
        PWMR.ChangeDutyCycle(100)
        PWML1.ChangeDutyCycle(100)
        time.sleep(0.8)
        stop()
            
'''

* Function Name:	direction
* Input:		a -> a list containing the nodes that the robot has to travel
* Output:		determines the next direction the robot has to take and
                        turns the robot accordingly 
* Logic:		compares the present node to the next node and determines
                        the direction by the position of next node
* Example Call:	        direction(a)
*
'''                
def direction(a):
    # i: the index of the array a
    # status: the current direction of the robot is stored in this variable
    i=0
    status='up'
    while i<len(a)-1:
        j=i+1
        if type(a[j])==str:
            package_client.Message(a[j])
        if type(a[j])==int and type(a[i])==str:
            if type(a[i-1])==int:
                a[i]=a[i-1]
            else:
                a[i]=a[i-2]
        if type(a[j])==int and type(a[i])==int:
            p=a[j]-a[i]
            if status=='right':
                if p==1:
                    path_following()        
                    status='right'
                elif p==7:
                    turn(3)
                    path_following()
                    status='down'
                elif p==-7:
                    turn(4)
                    path_following()
                    status='up'
                elif p==-1:
                    turn(3)
                    turn(3)
                    path_following()
                    status='left'
            elif status=='left':
                if p==-1:
                    path_following()
                    status='left'
                elif p==7:
                    turn(4)
                    path_following()
                    status='down'
                elif p==-7:
                    turn(3)
                    path_following()
                    status='up'
                elif p==1:
                    turn(3)
                    turn(3)
                    path_following()
                    status='right'
            elif status=='up':
                if p==1:
                    turn(3)
                    path_following()
                    status='right'
                elif p==7:
                    turn(3)
                    turn(3)
                    path_following()
                    status='down'
                elif p==-7:
                    path_following()
                    status='up'
                elif p==-1:
                    turn(4)
                    path_following()
                    status='left'
            elif status=='down':
                if p==1:
                    turn(4)
                    path_following()
                    status='right'
                elif p==-7:
                    turn(3)
                    turn(3)
                    path_following()
                    status='up'
                elif p==7:
                    path_following()
                    status='down'
                elif p==-1:
                    turn(3)
                    path_following()
                    status='left'
        i=j
'''

* Function Name:	path_following
* Input:		void-> this function takes no input arguments
* Output:		when called controls the motors of the robot by tracing
                        the black line until a node is encountered
* Logic:		the Raspberry Pi extracts the black line position by using
                        image processing techniques and controls motors
* Example Call:         path_following()
*
'''
def path_following():
    # last_dir: this is used to keep track of the movement of the robot
    last_dir=0
    cap=cv2.VideoCapture(0)
    while 1:
        hist=0
        #his : it is used to store the histogram values of the image
        ret, frame=cap.read()
        rows,cols,dim=frame.shape
        #converting the RGB image to HSV image
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lb=np.array([110,50,50])
        ub=np.array([130,255,255])
        mask=cv2.inRange(hsv,lb,ub)
        hist=cv2.calcHist([mask],[0],None,[1],[100,256])
        # to find whether the robot has reached a blue colored node
        if hist>70000:
            select(1)
            time.sleep(0.6)
            stop()
            cap.release()
            return
       
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # to trace the black line and fidn whether the robot is on the line or not
        mask=cv2.inRange(hsv,np.array([0,0,0]),np.array([255,255,180]))
        Right = mask[rows/2][cols-20]
        Left = mask[rows/2][20]
        Mid = mask[rows/2][cols/2]
        if Right==0 and Mid==255 and Left==0:
            select(1)
            time.sleep(0.2)
            stop()
            
        elif Right==255 and Mid==255:     
            select(3)
            time.sleep(0.1)
            last_dir='r'
            stop()
            
        elif Left==255 and Mid==255:
            select(4)
            time.sleep(0.1)
            last_dir='l'
            stop()
        elif Left==0 and Mid==0 and Right==0:
            if last_dir=='r':
                select(4)
                time.sleep(0.2)
                stop()
            else :
                select(3)
                time.sleep(0.2)
                stop()
        else:
            select(1)
            time.sleep(0.1)
            stop()
'''

* Function Name:	filter
* Input:		char-> the color of the package
                        area-> the area of the package
                        node-> a list containing the coordinates of the node
* Output:		determines the packages which are in pickup junctions area
                        and which are in deliver areas
* Logic:		checks the x coordinaes of the nodes.. If they are above
                        one sixth of the total height they are in deliver areas
                        else are in PUJ
* Example Call:	        filter('O',370,[2,3])
*
'''
def filter(color,area,node):
    # area: to find the shape of the contour with it's area
    if area > 370 and  area < 460 :
        area ='S'
    elif area > 90 and area < 150 :
        area ='T'
    elif area > 250 and area < 330 :
        area = 'C'
    else: return
    # to find the node of the package and appends it to either pickup or deliver arrays
    if node[0]<2:
        p=[7*node[0]+node[1],color+area]
        pickup.append(p)
    else :
        d=[7*node[0]+node[1],color+area]
        deliver.append(d)
        d_nodes.append(7*node[0]+node[1])
'''

* Function Name:	find_packages
* Input:	        image -> an image in the form of an numpy array
* Output:		forms a pickup and a deliver array containing the
                        color,shape and node of that package
* Logic:	        checks the contours for the four colors of the packages
                        and appends them eihter pick up or deliver array by calling
                        the filter function
* Example Call:	        find_packages(img)
*
'''        
def find_packages(frame):
    
    c=0
    height,width,dim=frame.shape
    z=height/5
    while c<4:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        arr1=[np.array([75,150,255]),np.array([145,255,255]),np.array([46,255,255]),np.array([0,201,0])]
        arr2=[np.array([141,255,255]),np.array([255,255,255]),np.array([72,255,255]),np.array([59,255,255])]
        mask = cv2.inRange(hsv,arr1[c],arr2[c])

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        kernel=np.ones((5,5),np.uint8)

        mask=cv2.erode(mask,kernel,iterations=1)
        contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        n= len(contours)
        p=0
        q=0
        mines_cont_bombs=[]
        # to find the coordinates of the package thorugh it's  centroid 
        while p<n:
            q=0
            M=cv2.moments(contours[p])
            if M['m00']>40 and M['m00']<1500:
                cx=int(M['m10']/M['m00'])
                cy=int(M['m01']/M['m00'])
                if cy>height/6:
                    i=(cy/50+1)/2
                    j=(cx/50+1)/2
                else:
                    i=  cy/100+1
                    j = cx/100+1
                cv2.drawContours(frame,contours,p,(0,0,255),3)
                area=cv2.contourArea(contours[p])
                if c==0:
                    color='B'
                elif c==1:
                    color='P'
                elif c==2:
                    color='G'
                elif c==3:
                    color='O'
                filter(color,area,[i,j])    
                
            p+=1
        c=c+1          
    mines_cont_bombs.sort()
    return mines_cont_bombs

'''

* Function Name:	pop
* Input:		prior -> It stores the distance of each node in shortest path
* Output:		lowest_key -> it is the variable containing the minimum distance among all the nodes in prior
* Logic:		for every distance in prior, it is compared with a variable (initially assigned as 1000)
*                       if it satisfies the condition then value of low and lowest_key is updated
*                       and finally returned the smallest distance
* Example Call:	        u = pop(prior)
*
'''
def pop(prior):
    # pop function pops out the element with lowest value.
    low = 1000
    lowest_key = None
    for key in prior:
        if prior[key] <= low:
            low = prior[key]    # low is updated
            lowest_key = key    # lowest_key is updated
    del prior[lowest_key]       # pop out the lowest value
    return lowest_key
'''
* Function Name:	dijkstra
* Input:		graph -> returned by function links2graph.It is a dictionary that displays the nodes connected to each node and their distances.
*	                start -> the starting point of the traversal in the graph
* Output:		returns the dist -> minimum distance for each nodes in the shortest path
*                       returns the pred -> It is a dictionary which stores the predecessor node of each node
* Logic:		initially every node except the starting node(have 0 as a distance) is assigned a higher value like 1000 as a distance
*			distance of each node is pushed to prior and and then poped in ascending order.
*                       newdist is formed adding the weights and then compared if it is shorter than previous distance. 
*			if found shorter saved it.
* Example Call:	        dist,pred=dijkstra(graph,start)
*
''' 
def dijkstra(graph, start):
    # to keep track of minimum distance from start to a vertex.
    prior = {}
    dist = {}   
    pred = {}   
 
    # initializing dictionaries
    for v in graph:
        dist[v] = 1000
        pred[v] = -1
    dist[start] = 0
    for v in graph:
        prior[v] = dist[v] # equivalent to push operation.
 
    while prior:
        u = pop(prior) # pop will get the element with smallest value
        for v in graph[u].keys(): # for each neighbor of u
            w = graph[u][v] # distance u to v
            newdist = dist[u] + w
            if (newdist < dist[v]): # check if new distance shorter than one in dist?
                # if found new shorter distance. save it
                prior[v] = newdist
                dist[v] = newdist
                pred[v] = u
 
    return dist, pred
'''

* Function Name:	links2graph
* Input:		horizontal_links -> It is a two-dimensional array and displays 1 or 0
*                       vertical_links -> It is a two-dimensional array and displays 1 or 0
*                       if links are present between two horizontal/vertical nodes or points in graph then it displays 1 otherwise 0
* Output:		graph -> It is a dictionary that displays the nodes connected to each node and their distances.
* Logic:		At a particular node , every horizontal and vertical links are checked.
*                       if found present, graph is made with corresponding distances.
* Example Call:	        graph=links2graph(horizontal_links,vertical_links)
*
'''
def links2graph(horizontal_links,vertical_links):
    graph={}
    inc=1
    x=0
    y=0
    while inc<=49:
        a=1000
        b=1000
        c=1000
        d=1000
        dup={}
        #  to check the adjacent links of a node from horizontal and vertical linsk matrices
        if x-1<7 and y<7 and x-1>=0 and y>=0 and vertical_links[x-1][y]==1:
            a = 7*(x-1)+y
            nextx=x-1
            nexty=y
        if x+1<7 and x+1>=0 and y<7 and y>=0 and vertical_links[x][y]==1:
            b = 7*(x+1)+y
            nextx=x+1
            nexty=y
        if x<7 and y-1<7 and x>=0 and y-1>=0 and horizontal_links[x][y-1]==1:
            c = 7*(x)+(y-1)
            nextx=x
            nexty=y-1
      
        if x<7 and y+1<7 and x>=0 and y+1>=0 and horizontal_links[x][y]==1:
            d = 7*(x)+y+1
            nextx=x
            nexty=y+1
        y=y+1
        if y==7:
            y=0
            x+=1

        if a!=1000:
            dup[a]=1
        if b!=1000:
            dup[b]=1
        if c!=1000:
            dup[c]=1
        if d!=1000:
            dup[d]=1
        graph[inc-1]=dup
        
        inc=inc+1
    return graph    
'''

* Function Name:	grid_to_arrays
* Input:		img-> a numpy array of an image
* Output:		returns two 2*2 lists containing the information whether the
                        links between two nodes are presernt or not
* Logic:		checks the pixels at each possible location of the links
                        if the pixels at that location are black then the link is
                        present else it is not 
* Example Call:	        grid_to_arrays(img)
*
'''
def grid_to_arrays(img):

    np_array1=np.zeros((7,6),np.uint8)
    horizontal_links=np_array1.tolist()

    np_array2=np.zeros((6,7),np.uint8)
    vertical_links=np_array2.tolist()
    
    rows,cols,dim=img.shape
    x=0
    s=rows/6
    s=s-1
    t=rows/12
    while x<=6:
        y=0
        while y<6:
                    p=img[x*s][t+y*s]
                    if all([q<40 for q in p]):
                        horizontal_links[x][y]=1
                    else:
                        horizontal_links[x][y]=0
                    y+=1
        x+=1
    x=0
    y=0
    while x<=6:
        y=0
        while y<6:
                    p=img[t+y*s][x*s]
                    if all([q<40 for q in p]):
                        vertical_links[y][x]=1
                    else:
                        vertical_links[y][x]=0
                    y+=1
        x+=1

    x=0
    while x<=6:
        y=0
        while y<6:
                    p=img[x*s][y*s]
                    if all([q==255 for q in p]):
                        horizontal_links[x][y-1]=0
                        horizontal_links[x][y]=0
                        vertical_links[x-1][y]=0
                        vertical_links[x][y]=0
                    y+=1
        x+=1

    return horizontal_links,vertical_links
'''

* Function Name:	call
* Input:		d -> contains the nodes e.g.-> d_nodes , contains the nodes of delivery positions
*                       start -> starting point in traversal 
* Output:		r is returned as the minimum distanced node from start
* Logic:	        if distance of ending node is equal to the minimum of distances appended,
*                       then path is updated and the last node is returned which will be at minimum distance
* Example Call:	        r=call(start,d_nodes)
*
'''
def call(start,d):
    dist, pred= dijkstra(graph,start)
    b=[]
    c=0
    r=0
    for v in d:
        b.append(dist[v])   #stores the distance of every nodes
    for x in d:
        end=x               #end is updated
        a=[]    
        if dist[end]==min(b):   #compared with minimum of b
            c=c+1
            if c>=2:
                continue
            a.append(end)
            while end!=start:
                a.append(pred[end])
                end=pred[end]
            a.reverse()
            path.append(a)  #path is updated
            r=x             #r is returned when dist[x]==min(b)
            d.remove(x)     #x is removed from d
                           
    return r

# to initialize the PWM modes on the GPIO pins of the raspberry pi
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(7,GPIO.OUT)
GPIO.setup(8,GPIO.OUT)

GPIO.setup(24,GPIO.OUT)
GPIO.setup(23,GPIO.OUT)
GPIO.setup(27,GPIO.OUT)
GPIO.setup(22,GPIO.OUT)

E1=GPIO.PWM(7,60)
E2=GPIO.PWM(8,60)
PWMR=GPIO.PWM(24,60)
PWMR1=GPIO.PWM(23,60)
PWML=GPIO.PWM(27,60)
PWML1=GPIO.PWM(22,60)

E1.start(0)
E2.start(0)
PWMR.start(0)
PWMR1.start(0)
PWML.start(0)
PWML1.start(0)

E1.ChangeDutyCycle(100)
E2.ChangeDutyCycle(100)
'''

* Function Name:	get_puj
* Input:		none
* Output:		g as array of pick_up_junctions
* Logic:	        compared the color and shape of every nodes in the path so that locations of parcels and PUJ can be extracted.
* Example Call:	        g=get_puj()
*
'''
def get_puj():
    count=0
    n=0
    start=42
    u=[]
    g=[]
    l=0
    while len(d_nodes)!=0:
        a=[]
        if count==4*n:      #after delivering 4 parcels to return, end is made 10
            n+=1
            end=10
            dist,pred=dijkstra(graph,start)     #gets shortest path from start to end
            a.append(end)
            while end!=start:
                a.append(pred[end])
                end=pred[end]
            a.reverse()
            path.append(a)
            start=10
        r=call(start,d_nodes)       #r is returned as the minimum distanced node from start
        start=r                     #start is updated
        count=count+1
    for x in path:                  #compares the color and shape of every nodes in the path with the pick_up_junction parcels
        if l==4:
            g.sort()                #sorted after comparing 4
        for y in deliver:
            if y[0]==x[-1]:
                for z in pickup:
                    if z[1]==y[1]:
                        g.append(z[0])
                        l+=1           
    return g
 



img=cv2.imread('CS_Original_Test_Image.jpg')
frame=cv2.resize(img,None,fx=0.1,fy=0.1,interpolation=cv2.INTER_CUBIC)
horizontal_links,vertical_links=grid_to_arrays(frame)
graph=links2graph(horizontal_links,vertical_links)
hor=find_packages(frame)
pickup.sort()
deliver.sort()
count=0
start=42
n=0
d_nodes_copy=[]
l=0
while l<len(d_nodes):
    d_nodes_copy.append(d_nodes[l])
    l=l+1
g=get_puj()
empty=[]
path=empty
count=0
temp=0
start=42
n=0
m=0
p=4
# to find the shortest distance from the first PUJ  to the packages and
# hence fidn the shortest path among the delivery junctions
while len(d_nodes_copy)!=0:
    a=[]
    if count==4*n:
        n+=1
        end=10
        dist,pred=dijkstra(graph,start)
        a.append(end)
        while end!=start:
            a.append(pred[end])
            end=pred[end]
        a.reverse()
        path.append(a)
        start=10
        if start==10:
            k=g[m:p]
            m=p
            p=p+4
            for q in k:
                a=[]
                b=[] 
                end=q
                dist,pred=dijkstra(graph,start)
                a.append(end)
                while end!=start:
                    a.append(pred[end])
                    end=pred[end]
                a.reverse()
                path.append(a)
                start=q
            end=10
            dist,pred=dijkstra(graph,start)
            b.append(end)
            while end!=start:
                b.append(pred[end])
                end=pred[end]
            b.reverse()
            path.append(b)
            start=10            
    r=call(start,d_nodes_copy)
    start=r
    count=count+1

puj=[]
dj=[]

length=len(path)
counter=0
while counter<length+len(deliver)-1:
    if path[counter][-1]>13:
        dj.append(path[counter][-1])
        for k in deliver:
            if path[counter][-1]==k[0]:
                path.insert(counter+1,[k[1]+'0'])
    else:
        puj.append(path[counter][-1])
        
    counter+=1

for i in dj:
    if type(i)!= int:
        dj.remove(i)

k=dj[0:4]
counter=0
n=0
# the loop to find the corresponding node of the package        
for x in path:
    if n==4 :
        k=dj[4:]
    for c in pickup:
        if c[0]==x[-1]:
            for t in k:
                for z in deliver:
                    
                    if t==z[0]:
                        
                        if c[1]==z[1]:
                            n+=1
                            if n==5:
                                continue
                            path.insert(counter+1,[c[1]+'1'])
                            k.remove(t)
    
        
    counter+=1
# to extract the nodes that the robot has to travel
for x in path:
    if type(x[0])==int:
        del x[0]
#route : to find the route map of the robot
route=[]
for x in path:
        route+=x
# to display the end of the program 
end=['PS1','GS1','BS1','OS1','ccc']
 
route=route+end
direction(route)    

stop()
GPIO.cleanup()
cv2.destroyAllWindows()
quit()
