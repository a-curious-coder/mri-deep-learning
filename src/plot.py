from os.path import exists

import os
import numpy as np
from plotly import express as px
from plotly import graph_objs as go
from skimage.transform import resize
from skimage.util import montage
# Import matplotlib
import matplotlib.pyplot as plt
import csv


def create_plot_folder(patient_id = None):
    """ Create folder for plots
    
    Args:
        patient_id (int): patient id
    """
    if patient_id is not None:
        if not exists(f"../plots/{patient_id}"):
            os.mkdir(f"../plots/{patient_id}")


def save_figure(fig, file_name, patient_id = None, file_type = "png"):
    """ Save figure to file
    
    Args:
        fig (matplotlib.figure): figure
    """
    # Save figure
    if patient_id is not None:
        create_plot_folder(patient_id)
        if file_type == "png":
            fig.savefig(f"../plots/{patient_id}/{file_name}.png")
        elif file_type == "pdf":
            fig.savefig(f"../plots/{patient_id}/{file_name}.pdf")
        elif file_type == "html":
            fig.write_html(f"../plots/{patient_id}/{file_name}.html")
    else:
        if file_type == "png":
            fig.savefig(f"../plots/{file_name}.png")
        elif file_type == "pdf":
            fig.savefig(f"../plots/{file_name}.pdf")
        elif file_type == "html":
            fig.write_html(f"../plots/{file_name}.html")


def smooth_curve(points, factor=0.8):
    """
    Smooths a curve by taking the average of the points.
    Args:
        points (list): points
        factor (float): smoothing factor
    Returns:
        list: smoothed points
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_history(history=None, guid=None):
    """Plot training history

    Args:
        history (keras.callbacks.History): training history
        epochs (int): number of epochs

    Returns:
        None
    """
    if history is None:
        if guid is None:
            guid = "9_5_32"
        # Load acc and val_acc from results
        acc_df = pd.read_csv(f"../data/results/acc_{guid}.csv")
        acc = acc_df["acc"].values
        val_acc = acc_df["val_acc"].values
        # Load loss and val_loss from results
        loss_df = pd.read_csv(f"../data/results/loss_{guid}.csv")
        loss = loss_df["loss"].values
        val_loss = loss_df["val_loss"].values
    else:
        acc = history["acc"]
        val_acc = history["val_acc"]
        loss = history["loss"]
        val_loss = history["val_loss"]

        # Save accuracies to csv file
        with open(f"../data/results/acc_{guid}.csv", "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "acc", "val_acc"])
            for i in range(len(acc)):
                writer.writerow([i, acc[i], val_acc[i]])

        # Save losses to csv file
        with open(f"../data/results/loss_{guid}.csv", "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "val_loss"])
            for i in range(len(loss)):
                writer.writerow([i, loss[i], val_loss[i]])

    # Generate list of epochs
    epochs = []
    for i in range(1, len(acc)+1):
        epochs.append(i)

    # Make sure folder epochs_{epochs} exists
    if not exists(f"../plots/epochs_{len(epochs)}"):
        os.mkdir(f"../plots/epochs_{len(epochs)}")

    # Plot training and validation accuracy using Plotly
    fig = go.Figure(data=[
            go.Scatter(x=epochs,
                        y=smooth_curve(acc),
                        mode="markers",
                        name="Training Accuracy",
                        marker=dict(size=10)),
            go.Scatter(x=epochs,
                        y=smooth_curve(val_acc),
                        mode="lines",
                        line=dict(width=2),
                        name="Validation Accuracy")
    ])
    # Update layout of plot
    fig.update_layout(title="Training and Validation Accuracy",
                        width=800,
                        xaxis=dict(showgrid=False,
                                    title="Epochs"),
                        yaxis=dict(showgrid=False,
                                    title="Accuracy"))
    # Save plot to file
    fig.write_image(f"../plots/epochs_{max(epochs)}/{guid}_acc.png")


    
    # Plot training and validation loss using Plotly
    fig = go.Figure(data=[
            go.Scatter(x=epochs,
                        y=smooth_curve(loss),
                        mode="markers",
                        name="Training Loss",
                        marker=dict(size=10)),
            go.Scatter(x=epochs,
                        y=smooth_curve(val_loss),
                        mode="lines",
                        name="Validation Loss",
                        line=dict(width=2))
    ])
    fig.update_layout(title="Training and Validation Loss",
                        width=800,
                        xaxis=dict(showgrid=False,
                                    title="Epochs"),
                        yaxis=dict(showgrid=False,
                                    title="Loss"))
    fig.write_image(f"../plots/epochs_{max(epochs)}/{guid}_loss.png")

    print(f"[INFO] Saved acc/loss plots to ../plots/epochs_{max(epochs)}")


def plot_mri_slices(image, label, patient_id=None, slice_mode = "center", show = False):
    """ Plots each slice of the MRI image
    Slices are plotted in this order: Saggital, Coronal and Axial

    Args:
        image (np array): image
        label (str): label
    """
    image = np.array(image)
    print(f"[INFO] Patient {patient_id:<5} {slice_mode:<5} slices")
    # Plot each channel separately
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    axs[0].imshow(image[:, :, 0], cmap="gray")
    axs[0].set_title("Saggital")
    axs[1].imshow(image[:, :, 1], cmap="gray")
    axs[1].set_title("Coronal")
    axs[2].imshow(image[:, :, 2], cmap="gray")
    axs[2].set_title("Axial")

    # remove axis for all frames
    for frame in axs:
        frame.axis("off")

    # Tight layout
    fig.tight_layout()

    diagnosis = "Alzheimer's" if label == 1 else "Non-Alzheimer's"
    diagnosis = label if label != 0 else "Non-Alzheimer's"
    suptitle = f"{patient_id:<5} {diagnosis} {slice_mode:<5}" 
    fig.suptitle(suptitle, fontsize=20)

    save_figure(fig, f"{label}_{slice_mode}_slices", patient_id, "png")

    if show:
        plt.show()


def plot_all_frames(test_image_t1):
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    ax1.imshow(
        rotate(montage(test_image_t1[50:-50, :, :]), 90, resize=True), cmap='gray')


def plot_mri_slice(patient_id, patient_diagnosis, image, directory="plots"):
    """ Plots MRI image slice

    Args:
        patient_id (str): Patient ID
        patient_diagnosis (str): Patient diagnosis
        image (np.array): MRI image
        directory (str): Directory to save plot

    Returns:
        None
    """

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
    fig.show()
    # fig.write_image(
    #     f'{directory}/{patient_diagnosis}/{patient_id}.png')
    return True


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
