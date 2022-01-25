import plotly.graph_objs as go
from plotly.subplots import make_subplots

def verify_import():
    print("[*]\tvisualisations.py successfully imported")
    pass

def plot_loss(loss, val_loss):
    epochs_range = list(range(1, len(loss)+1))
    loss = [i*100 for i in loss]
    val_loss = [i*100 for i in val_loss]
    training_loss = go.Scatter(
        x = epochs_range,
        y = loss,
        mode = 'markers',
        name = 'Training Loss',
        marker = dict(color = 'red'),
        hovertemplate="Epoch %{x}<br>Training Loss: %{y:.2f}%<extra></extra>"
    )
    
    validation_loss = go.Scatter(
        x = epochs_range,
        y = val_loss,
        mode = 'markers+lines',
        name = "Validation Loss",
        marker = dict(color = 'blue'),
        hovertemplate="Epoch %{x}<br>Validation Loss: %{y:.2f}%<extra></extra>"
    )

    data = [training_loss, validation_loss]
    layout = go.Layout(
        # yaxis = dict(range = (0, 100)),
        title = dict(text = 'Training/Validation Loss', x = 0.5)
    )
    fig = go.Figure(data=data, layout = layout)
    fig.write_html('plots/mri_loss.html')
    return fig

def plot_accuracy(acc, val_acc):
    """Plots accuracy statistics for neural net models

    Args:
        acc ([type]): [description]
        val_acc ([type]): [description]

    Returns:
        [type]: [description]
    """
    epochs_range = list(range(1, len(acc)+1))
    acc = [i*100 for i in acc]
    val_acc = [i*100 for i in val_acc]
    training_acc = go.Scatter(
        x = epochs_range,
        y = acc,
        mode = 'markers',
        name = 'Training Accuracy',
        marker = dict(color = 'red'),
        hovertemplate="Epoch %{x}<br>Training accuracy: %{y:.2f}%<extra></extra>"
    )
    validation_acc = go.Scatter(
        x = epochs_range,
        y = val_acc,
        mode = 'markers+lines',
        name = "Validation Accuracy",
        marker = dict(color = 'blue'),
        hovertemplate="Epoch %{x}<br>Validation accuracy: %{y:.2f}%<extra></extra>"
    )
    data = [training_acc, validation_acc]
    layout = go.Layout(
        yaxis = dict(range = (0, 100)),
        title = dict(text = 'Training/Validation Accuracy', x = 0.5)
    )
    fig = go.Figure(data=data, layout = layout)
    fig.write_html('plots/mri_accuracy.html')
    return fig

def plot_all_graphs(loss, val_loss, acc, val_acc):
    """Plots the neural network model's accuracy and loss on same figure
    Args:
        loss ([type]): [description]
        val_loss ([type]): [description]
        acc ([type]): [description]
        val_acc ([type]): [description]
    """
    fig = make_subplots(rows=1, cols=2)

    # Get loss and accuracy statistics
    fig1 = plot_loss(loss, val_loss)
    fig2 = plot_accuracy(acc, val_acc)

    fig.append_trace(fig1, row = 1, col = 1)
    fig.append_trace(fig2, row = 1, col = 2)

    fig.update_layout(height = 600, width = 1000, title = dict(text = "Deep Learning Model Statistics", x = 0.5))
    

