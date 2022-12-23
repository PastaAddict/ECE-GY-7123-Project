import torch
from torch import nn
import numpy as np

def fast_adapt_poisson(task, boundary, learner, adaptation_steps, device):
    lossfn = nn.MSELoss(reduction='mean')

    # Separate data into adaptation/evalutation sets
    train_inputs, train_targets = torch.from_numpy(task[:][0]).float().to(device), torch.from_numpy(task[:][1]).float().to(device)
    x_support, f_support = train_inputs[::2,:2], train_targets[::2].reshape(-1,1)
    x_query, f_query = train_inputs[1::2,:2], train_targets[1::2].reshape(-1,1)
    x_boundary, g_boundary = boundary[0], boundary[1]

    

    # Adapt the model
    for step in range(adaptation_steps):
        x = x_support.clone()
        x.requires_grad = True
        ''' 
        Boundary condition fitting
        Calculating the loss that expresses how well our solution respects the boundary conditions
        '''
        u_boundary = learner(x_boundary)
        support_loss_bc=lossfn(u_boundary, g_boundary)

        ''' 
        PDE fitting
        Calculating the loss that expresses how well our solution respects the ODE
        u_xx + u_yy = f(x,y)  
        '''
        u = learner(x) #u
        du = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
        u_x = du[:,0].reshape(-1,1) #u_x
        du_x = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0] 
        u_xx = du_x[:,0].reshape(-1,1) #u_xx
        u_y  = du[:,1].reshape(-1,1) #u_t
        du_y = torch.autograd.grad(u_y.sum(),x,create_graph=True)[0] 
        u_yy = du_y[:,1].reshape(-1,1) #u_xx
        support_loss_r=lossfn(u_xx+u_yy,f_support)

        ''' Adapt the model by optimizing the sum of the two losses, Boundary condition loss + ODE loss'''
        learner.adapt(support_loss_r+support_loss_bc)
    #print(support_loss_bc, support_loss_r)

    # Evaluate the adapted model
    ''' Repeat the same process to calculate the meta train loss '''
    x = x_query.clone()
    x.requires_grad = True

    ''' Boundary conditions '''
    u_boundary = learner(x_boundary)
    query_loss_bc=lossfn(u_boundary, g_boundary)
    
    ''' PDE '''
    u = learner(x) #u
    du = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    u_x = du[:,0].reshape(-1,1) #u_x
    du_x = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0] 
    u_xx = du_x[:,0].reshape(-1,1) #u_xx
    u_y  = du[:,1].reshape(-1,1) #u_t
    du_y = torch.autograd.grad(u_y.sum(),x,create_graph=True)[0] 
    u_yy = du_y[:,1].reshape(-1,1) #u_xx
    query_loss_r=lossfn(u_xx+u_yy,f_query)

   


    return query_loss_r + query_loss_bc


def fast_adapt_integral(task, boundary, learner, adaptation_steps, device):
    lossfn = nn.MSELoss(reduction='mean')

    # Separate data into adaptation/evalutation sets
    train_inputs, train_targets = torch.from_numpy(task[:][0]).float().to(device), torch.from_numpy(task[:][1]).float().to(device)
    x_support, f_support = train_inputs[::2], train_targets[::2]
    x_query, f_query = train_inputs[1::2], train_targets[1::2]
    x_boundary, g_boundary = boundary[0], boundary[1]

    

    # Adapt the model
    for step in range(adaptation_steps):
        x = x_support.clone()
        x.requires_grad = True
        ''' 
        Initial condition fitting
        Calculating the loss that expresses how well our solution respects the initial conditions
        '''
        u_boundary = learner(x_boundary)
        support_loss_bc=lossfn(u_boundary, g_boundary)

        ''' 
        ODE fitting
        Calculating the loss that expresses how well our solution respects the ODE
        u' + u = f
        '''
        u = learner(x) #u
        u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0] #u'
        support_loss_r=lossfn(u_x, f_support) 

        ''' Adapt the model by optimizing the sum of the two losses, Boundary condition loss + ODE loss'''
        learner.adapt(support_loss_r+support_loss_bc)

    # Evaluate the adapted model
    ''' Repeat the same process to calculate the meta train loss '''
    ''' Initial conditions '''
    x = x_query.clone()
    x.requires_grad = True

    u_boundary = learner(x_boundary)
    query_loss_bc=lossfn(u_boundary, g_boundary)
    
    ''' ODE '''
    u = learner(x)
    u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0]

    query_loss_r = lossfn(u_x, f_query)
   


    return query_loss_r + query_loss_bc


def fast_adapt_heat(task, boundary, learner, adaptation_steps, device, residual):
    k=0.1
    lossfn = nn.MSELoss(reduction='mean')

    # Separate data into adaptation/evalutation sets
    train_inputs, train_targets = torch.from_numpy(task[:][0]).float().to(device), torch.from_numpy(task[:][1]).float().to(device)
    x_support, f_support = train_inputs[::2,:2], train_targets[::2]
    x_query, f_query = train_inputs[1::2,:2], train_targets[1::2]
    x_boundary, g_boundary = boundary[0], boundary[1]
    f_support = f_support*torch.sin(x_support[:,0]*np.pi)
    f_support = f_support.reshape(-1,1)
    f_query = f_query*torch.sin(x_query[:,0]*np.pi)
    f_query = f_query.reshape(-1,1)

    

    # Adapt the model
    for step in range(adaptation_steps):
        xt = residual.clone()
        xt.requires_grad = True
        ''' 
        Boundary condition fitting
        Calculating the loss that expresses how well our solution respects the boundary conditions
        '''
        u_boundary = learner(x_boundary)
        support_loss_bc=lossfn(u_boundary, g_boundary)

        # Boundary condition as defined by the specific task
        u_boundary_task = learner(x_support)
        
        support_loss_bc += lossfn(u_boundary_task, f_support)

        ''' 
        PDE fitting
        Calculating the loss that expresses how well our solution respects the ODE
        ku_xx = u_t 
        '''
        u = learner(xt) #u
        du = torch.autograd.grad(u.sum(),xt,create_graph=True)[0]
        u_x = du[:,0].reshape(-1,1) #u_x
        du_x = torch.autograd.grad(u_x.sum(),xt,create_graph=True)[0] 
        u_xx = du_x[:,0].reshape(-1,1) #u_xx
        u_t  = du[:,1].reshape(-1,1) #u_t
        support_loss_r=lossfn(k*u_xx,u_t) 

        ''' Adapt the model by optimizing the sum of the two losses, Boundary condition loss + ODE loss'''
        learner.adapt(support_loss_bc+support_loss_r)
        #print(support_loss_r ,support_loss_bc)

    # Evaluate the adapted model
    ''' Repeat the same process to calculate the meta train loss '''
    xt = residual.clone()
    xt.requires_grad = True

    ''' Boundary conditions '''
    u_boundary = learner(x_boundary)
    query_loss_bc=lossfn(u_boundary, g_boundary)

    # Boundary condition as defined by the specific task
    u_boundary_task = learner(x_query)
    query_loss_bc += lossfn(u_boundary_task, f_query)
    
    ''' ODE '''
    u = learner(xt) #u
    du = torch.autograd.grad(u.sum(),xt,create_graph=True)[0]
    u_x = du[:,0].reshape(-1,1) #u_x
    du_x = torch.autograd.grad(u_x.sum(),xt,create_graph=True)[0] 
    u_xx = du_x[:,0].reshape(-1,1) #u_xx
    u_t  = du[:,1].reshape(-1,1) #u_t
    query_loss_r=lossfn(k*u_xx,u_t)

    #print(query_loss_r ,query_loss_bc)
   


    return query_loss_r + query_loss_bc