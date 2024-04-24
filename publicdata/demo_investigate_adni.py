import os

root_dataset = "/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data" #"/home/jyn/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data"
list_scans = os.listdir(root_dataset)
list_subjects = []
for ith_scan in list_scans:
    if '.vtk' in ith_scan and 'sub' in ith_scan:
        list_subjects.append(int(ith_scan.split('_')[0][-4::]))

import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

# Reading the data inside the xml
# file to a variable under the name
# data
path ='/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data_set.xml'
with open(path, 'r') as f:
    data = f.read()

# Passing the stored data inside
# the beautifulsoup parser, storing
# the returned object
Bs_data = BeautifulSoup(data, "xml")

# Finding all instances of tag
# `unique`
b_unique = Bs_data.find_all('unique')

print(b_unique)

# Using find() to extract attributes
# of the first instance of the tag
b_name = Bs_data.find('child', {'name': 'Frank'})

print(b_name)


# Parse the XML file
tree = ET.parse('/Users/jyn/jyn/research/projects/NAISR/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/3d/hippocampi/data_set.xml')  # Replace with your XML file path
root = tree.getroot()
a = []
b = []
# Access elements and attributes
for child in root:
    print(child.tag, child.attrib)
    a.append(child.attrib)

# Access specific element data
for element in root.findall('your_element'):  # Replace 'your_element' with the desired element's tag name
    value = element.text
    print(value)
