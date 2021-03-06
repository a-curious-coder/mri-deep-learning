from os.path import exists

import numpy as np
from plotly import express as px
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from skimage.transform import resize
from skimage.util import montage


def verify_import():
    print("[*]\tvisualisations.py successfully imported")


def plot_history(history, epochs):
    """ Plots CNN history accummulated during training

    Args:
        history (??): Model training history
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history.history['loss'],
                             name='Training Loss',
                             marker_color='red'))
    fig.add_trace(go.Scatter(x=epochs, y=history.history['val_loss'],
                             name='Validation Loss',
                             marker_color='blue'))
    fig.add_trace(go.Scatter(x=epochs, y=history.history['sparse_categorical_accuracy'],
                             name='Training Accuracy',
                             marker_color='green'))
    fig.add_trace(go.Scatter(x=epochs, y=history.history['val_sparse_categorical_accuracy'],
                             name='Validation Accuracy',
                             marker_color='orange'))
    fig.update_layout(height=600, width=1000, title=dict(
        text="Training and Validation Loss and Accuracy",
        x=0.5,
        y=0.9,
        xanchor="center",
        yanchor="top"))
    saved = False
    num = 1
    name = f"../plots/history_{num}.png"
    while not saved:
        if not exists(name):
            fig.write_image(name)
            saved = True
        num += 1
        name = f"../plots/history_{num}.png"


def plot_loss(loss, val_loss):
    epochs_range = list(range(1, len(loss)+1))
    loss = [i*100 for i in loss]
    val_loss = [i*100 for i in val_loss]
    training_loss = go.Scatter(
        x=epochs_range,
        y=loss,
        mode='markers',
        name='Training Loss',
        marker=dict(color='red'),
        hovertemplate="Epoch %{x}<br>Training Loss: %{y:.2f}%<extra></extra>"
    )

    validation_loss = go.Scatter(
        x=epochs_range,
        y=val_loss,
        mode='markers+lines',
        name="Validation Loss",
        marker=dict(color='blue'),
        hovertemplate="Epoch %{x}<br>Validation Loss: %{y:.2f}%<extra></extra>"
    )

    data = [training_loss, validation_loss]
    layout = go.Layout(
        # yaxis = dict(range = (0, 100)),
        title=dict(text='Training/Validation Loss', x=0.5)
    )
    fig = go.Figure(data=data, layout=layout)
    fig.write_html('plots/mri_loss.html')
    return fig


def plot_accuracy(acc, val_acc, file_name):
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
        x=epochs_range,
        y=acc,
        mode='markers',
        name='Training Accuracy',
        marker=dict(color='red'),
        hovertemplate="Epoch %{x}<br>Training accuracy: %{y:.2f}%<extra></extra>"
    )
    validation_acc = go.Scatter(
        x=epochs_range,
        y=val_acc,
        mode='markers+lines',
        name="Validation Accuracy",
        marker=dict(color='blue'),
        hovertemplate="Epoch %{x}<br>Validation accuracy: %{y:.2f}%<extra></extra>"
    )
    data = [training_acc, validation_acc]
    layout = go.Layout(
        yaxis=dict(range=(0, 100)),
        title=dict(text='Training/Validation Accuracy', x=0.5)
    )
    fig = go.Figure(data=data, layout=layout)
    fig.write_html(f'plots/{file_name}.html')
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

    fig.append_trace(fig1, row=1, col=1)
    fig.append_trace(fig2, row=1, col=2)

    fig.update_layout(height=600, width=1000, title=dict(
        text="Deep Learning Model Statistics", x=0.5))


def plot_all_frames(test_image_t1):
    print("ok")
    # fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    # ax1.imshow(
    #     rotate(montage(test_image_t1[50:-50, :, :]), 90, resize=True), cmap='gray')


def plot_mri_slice(patient_id, patient_diagnosis, image, directory="plots"):
    print(type(image))
    if exists(f'{directory}/{patient_diagnosis}/{patient_id}.png'):
        return False
    fig = go.Figure(go.Image(z=image))
    fig = resize(image, (250, 250),
                 order=1, preserve_range=True)
    fig = px.imshow(image,
                    binary_string=True)
    # fig.update_layout(coloraxis_showscale=False)
    # fig.update_xaxes(showticklabels=False)
    # fig.update_yaxes(showticklabels=False)
    # fig.update_layout(
    #     title_text=f"{patient_id} MRI Image : {patient_diagnosis}",
    #     title_x=0.5,
    # autosize=False,
    # margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
    # )

    fig.write_image(
        f'{directory}/{patient_diagnosis}/{patient_id}.png')
    return True
    # fig.show()


def plot_mri_image(patient_id, patient_diagnosis, image):
    r, c = image[0].shape
    nb_frames = 256

    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    z=(25.6 - k * 0.1) * np.ones((r, c)),
                    surfacecolor=np.flipud(image[255 - k]),
                    cmin=0,
                    cmax=255
                ),
                # you need to name the frame for the animation to behave properly
                name=f"Slice {k}"
            )
            for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(
        go.Surface(
            z=25.6 * np.ones((r, c)),
            surfacecolor=np.flipud(image[255]),
            colorscale='Gray',
            cmin=0,
            cmax=255,
            # colorbar=dict(thickness=20, ticklen=4)
        ))

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[frame.name], frame_args(0)],
                    "label": str(count),
                    "method": "animate",
                }
                for count, frame in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title=dict(text=f"{patient_id} MRI Image : {patient_diagnosis}",
                   x=0.5),
        scene=dict(
            zaxis=dict(range=[-0.1, 25.6], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(10)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )
    fig.show()


def plot_mri_comparison(patient_images, patient_details):
    # 3, 256, 240, 160
    patient_images = np.asarray([image[128, :, :] for image in patient_images])
    print(type(patient_images))
    print(type(patient_images[0]))
    print(patient_images[0].shape)
    fig = px.imshow(patient_images,
                    facet_col=0,
                    binary_string=True,
                    facet_col_wrap=len(patient_images),
                    facet_col_spacing=0,
                    facet_row_spacing=0)
    fig.write_image("plots/mri_comparison.png", scale=5)
