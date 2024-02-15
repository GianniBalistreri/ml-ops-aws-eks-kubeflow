import plotly.graph_objects as go
import numpy as np
np.random.seed(1)

N = 100
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
sz = np.random.rand(N) * 30

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="markers",
    marker=go.scatter.Marker(
        size=sz,
        color=colors,
        opacity=0.6,
        colorscale="Viridis"
    )
))

#fig.write_image('fig1.png')

import io
import boto3

# Save the output to a Bytes IO object
png_data = io.BytesIO()
fig.write_image(png_data)
# Seek back to the start so boto3 uploads from the start of the data
#bits.seek(0)
png_data.seek(0)

# Upload the data to S3
s3 = boto3.client('s3')
s3.put_object(Bucket="shopware-ml-ops-interim-prod", Key="projects/avocado_price_prediction/results.jpg", Body=png_data)
