from data_visualizer import DataVisualizer
import json

with open('/Users/giannibalistreri/PycharmProjects/data_pipeline/feature_importance.json', 'r') as file:
    subplots = json.load(file)

viz = DataVisualizer(subplots=subplots)
p = viz.run()

for i, plot in enumerate(viz.subplots.keys()):
    if i == 0:
        continue
    print(viz.subplots[plot]['data'])
    break
