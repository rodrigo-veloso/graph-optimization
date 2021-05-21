import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from ortools.linear_solver import pywraplp
import time
import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from pyswarm import pso
from matplotlib.animation import FuncAnimation
import streamlit.components.v1 as components
import random

def plot_graph(G):
	plt.subplot(122)
	plt.box(False)
	plt.rcParams["figure.figsize"] = (10,3)
	pos=nx.spring_layout(G)
	nx.draw_networkx(G,pos)
	labels = nx.get_edge_attributes(G,'weight')
	nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
	st.pyplot(plt,width=5)
	plt.clf()

def calc_cost(G):
    cost = 0
    for weight in G.edges.data('weight'):
        cost += weight[-1]
    return cost

G = nx.Graph()

G.add_edge(0,1,weight=5)
G.add_edge(1,2,weight=1)
G.add_edge(2,3,weight=2)
G.add_edge(3,4,weight=4)
G.add_edge(4,1,weight=3)
G.add_edge(4,0,weight=1)
G.add_edge(3,1,weight=1)

#c = [5,3,2,4,3,1]
n = 5
d = 2*7

nodes = [i for i in range(n)]

for node in nodes: 
	G.add_node(node)

st.write("""## Grafo original""")

plot_graph(G)

st.write("""## Árvore geradora mínima, kruskal""")
start = time.time()
mst = nx.minimum_spanning_tree(G)
st.write("""### Tempo: {}""".format(time.time() - start))
cost = calc_cost(mst)
st.write("""### Custo: {}""".format(cost))

plot_graph(mst)

st.write("""## Árvore geradora mínima, programação inteira""")
model = pywraplp.Solver('mst',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

#[model.BoolVar('atendimento[%i,%i]' % (i, j)) for i in range(d)]
z = {}
c = {}
for i in range(n):
    for j in G.adj[i]:
        c[i,j] = G.adj[i][j]['weight']
        z[i,j] = model.BoolVar('[%i,%i]' % (i, j))
print(z)
print(c)
#teste = [(i, j)) for j in G.adj[i] for i in range(d)]
#teste = [[(i,j) for j in range(3)] for i in range(3)]
V = [model.IntVar(0, n-1, 'V[%i]' % (i)) for i in range(n)]

model.Minimize(model.Sum([c[tupla]*z[tupla] for tupla in z.keys()]))

model.Add(V[0] == 0)

for k in range(1,n):
    for i in G.adj[k]:
        model.Add(V[k] - V[i] - z[i,k] + (n - 2)*(1 - z[i,k]) - (n - 3)*z[k,i] >= 0)

for k in range(1,n):
    model.Add(V[k] >= 1)

for k in range(1,n):
    if (0,k) in z: 
        #st.write(z[0,k])   
        model.Add(V[k] - n + 1 + (n - 2)*z[0,k] <= 0)
    else:
        model.Add(V[k] - n + 1 <= 0)

for k in range(1,n):
    model.Add(model.Sum([z[i,k] for i in G.adj[k]]) == 1)

model.Add(model.Sum(z[0,k] for k in G.adj[0]) >= 1)

st.write('Número de restrições: {}'.format(model.NumConstraints()))

st.write('Número de variáveis: {}'.format(model.NumVariables()))

model.EnableOutput()

start = time.time()
model.Solve()
st.write("""### Tempo: {}""".format(time.time() - start))

st.write('Custo: {}'.format(model.Objective().Value()))

mst_model = nx.Graph()

nodes = [i for i in range(n)]

for node in nodes: 
    mst_model.add_node(node)

for i in range(n):
    for j in G.adj[i]:
        if z[i,j].solution_value() == 1:
        #st.write('aresta ({},{}): {}'.format(i,j,z[i,j].solution_value()))
            mst_model.add_edge(i,j,weight=c[i,j])

plot_graph(mst_model)

st.write("""## Árvore geradora mínima, algorítmo genético""")

def f(z):
    penalty = 0
    if z[d] != 0:
        penalty += 1000
    for k in range(1,n):
        for i in G.adj[k]:
            value = z[d+k] - z[d+i] - z[dic_index[i,k]] + (n - 2)*(1 - z[dic_index[i,k]]) - (n - 3)*z[dic_index[k,i]]
            if not (value >= 0):
                penalty += -100*value 
    for k in range(1,n):
        if z[d+k] == 0:
            penalty += 100
    for k in range(1,n):
        if (0,k) in dic_index: 
            #st.write(z[0,k])   
            value = z[d+k] - n + 1 + (n - 2)*z[dic_index[0,k]]
            if not value <= 0:
                penalty += 100*value
    for k in range(1,n):
        value = np.sum([z[dic_index[i,k]] for i in G.adj[k]])
        if not value == 1:
            penalty += 1000*np.abs(value - 1)
    value = np.sum(z[dic_index[0,k]] for k in G.adj[0])
    if not value >= 1:
        penalty += 1000 - 1000*value
    value = np.sum(z[:d])
    if not value == n - 1:
        penalty = 1000*np.abs(value - (n-1))
    return np.sum(c*z[:d]) + penalty

dic_index = {}
c = []
k = 0
for i in range(n):
    for j in G.adj[i]:
        dic_index[i,j] = k
        c.append(G.adj[i][j]['weight'])
        k += 1
c = np.array(c)
print(dic_index)

var_bound_z = [[0,1] for i in range(d)]
var_bound_V = [[0,n-1] for i in range(n)]
for bound in var_bound_V:
    var_bound_z.append(bound)
var_bound = np.array(var_bound_z)
model = ga(f, dimension = d+n, variable_type='int', variable_boundaries = var_bound, variable_type_mixed = None, function_timeout = 200,
                     algorithm_parameters = {'population_size':200}) 

start = time.time()
model.run()
st.write("""### Tempo: {}""".format(time.time() - start))

z = model.output_dict['variable']

mst_ga= nx.Graph()

nodes = [i for i in range(n)]

for node in nodes: 
    mst_ga.add_node(node)

for i in range(n):
    for j in G.adj[i]:
        if int(z[dic_index[i,j]]) == 1:
        #st.write('aresta ({},{}): {}'.format(i,j,z[i,j].solution_value()))
            mst_ga.add_edge(i,j,weight=c[dic_index[i,j]])

cost = calc_cost(mst_ga)
st.write("""### Custo: {}""".format(cost))
plot_graph(mst_ga)

st.write("""## Descida do gradiente""")
#ln, = ax.scatter([], [], 'r-')

def update1(frame,data,scatter):
    x0 = data[-1][0] - eta*2*data[-1][0]
    y0 = data[-1][1] - eta*2*data[-1][1]
    #x0 = 2*data[-1][0] + 20*np.pi*np.sin(2*np.pi*data[-1][0])
    #y0 = 2*data[-1][1] + 20*np.pi*np.sin(2*np.pi*data[-1][1])
    data.append([x0,y0])
    print(x0)
    scatter.set_offsets([x0,y0])
    return scatter,

def update2(frame,data,scatter):
    #x0 = data[-1][0] - eta*2*data[-1][0]
    #y0 = data[-1][1] - eta*2*data[-1][1]
    x0 = data[-1][0] - eta*(2*data[-1][0] + 20*np.pi*np.sin(2*np.pi*data[-1][0]))
    y0 = data[-1][1] = eta*(2*data[-1][1] + 20*np.pi*np.sin(2*np.pi*data[-1][1]))
    data.append([x0,y0])
    print(x0)
    scatter.set_offsets([x0,y0])
    return scatter,

def g(X):
    A = 10
    n = 2
    return A*n + (X[0]**2 - 10*np.cos(2*np.pi*X[0])) + (X[1]**2 - 10*np.cos(2*np.pi*X[1]))

def get_best(particles):
    best = 101
    for particle in particles:
        if particle.best < best:
            best = particle.best
            G = particle.X 
    return G

def update3(frame,data,scatter):
    #x0 = data[-1][0] - eta*2*data[-1][0]
    #y0 = data[-1][1] - eta*2*data[-1][1]
    #x0 = data[-1][0] - eta*(2*data[-1][0] + 20*np.pi*np.sin(2*np.pi*data[-1][0]))
    #y0 = data[-1][1] = eta*(2*data[-1][1] + 20*np.pi*np.sin(2*np.pi*data[-1][1]))
    #data.append([x0,y0])
    #print(x0)
    G = get_best(data)
    pos = []
    for particle in data:
        particle.update(G,g)
        pos.append(particle.X)
    #print(pos)
    scatter.set_offsets(pos)
    return scatter,


xlist = np.linspace(-5.12, 5.12, 100)
ylist = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = X**2 + Y**2
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z,levels=14, cmap="RdBu_r")
fig.colorbar(cp) # Add a colorbar to a plot
#ax.set_xlabel('x (cm)')
x0 = 4
y0 = 4
eta = 0.1
data1 = [[x0,y0]]
scatter = ax.scatter([],[],c='m')

ani = FuncAnimation(fig, update1, 25,fargs=(data1,scatter), interval = 100, blit=True)
components.html(ani.to_jshtml(), height=400)

#pĺt.clf()

X, Y = np.meshgrid(xlist, ylist)
A = 10
n = 2
Z = A*n + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z,levels=14, cmap="RdBu_r")
fig.colorbar(cp) # Add a colorbar to a plot
#ax.set_xlabel('x (cm)')
x0 = 4
y0 = 4
eta = 0.001
data2 = [[x0,y0]]
scatter = ax.scatter([],[],c='m')

ani = FuncAnimation(fig, update2, 25,fargs=(data2,scatter), interval = 100, blit=True)
components.html(ani.to_jshtml(), height=400)

st.write("""## PSO""")

class Particle():
    def __init__(self):
        self.X = 8*np.random.rand(2) - 4
        self.V = .2*np.random.rand(2) - .1
        self.P = self.X
        self.best = 100

    def update(self,G,f):
        self.V = 0.6*self.V + 2*random.random()*(self.P-self.X) + 2*random.random()*(G-self.X)
        self.X = self.X + self.V
        if f(self.X) < self.best:
            self.best = f(self.X)
            self.P = self.X        

X, Y = np.meshgrid(xlist, ylist)
A = 10
n = 2
Z = A*n + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z,levels=14, cmap="RdBu_r")
fig.colorbar(cp) # Add a colorbar to a plot
#ax.set_xlabel('x (cm)')
x0 = 4
y0 = 4
eta = 0.001
N = 10
data2 = [Particle() for i in range(N)]
for data in data2:
    print(data.X)
scatter = ax.scatter([],[],c='m')

ani = FuncAnimation(fig, update3,fargs=(data2,scatter), interval = 500, blit=True)
components.html(ani.to_jshtml(), height=400)
