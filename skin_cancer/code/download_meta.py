# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 00:25:07 2019

@author: lauri
"""

from isic_api import ISICApi
#Insert Username and Password Below
api = ISICApi(username="lrieger", password="UKn7@7f31qUZ")
#%%

#%%
import urllib
import os


imageList = api.getJson('image?limit=106&offset=0&sort=name')
#%%
            
            
print('Fetching metadata for %s images' % len(imageList))
imageDetails = []
for image in imageList:
    print(' ', image['name'])
    # Fetch the full image details
    imageDetail = api.getJson('image/%s' % image['_id'])
    imageDetails.append(imageDetail)
#%%
# Determine the union of all image metadata fields
metadataFields = set(
        field
        for imageDetail in imageDetails
        for field in imageDetail['meta']['clinical'].keys()
    )
#%%
import csv
metadataFields = ['isic_id'] + sorted(metadataFields)
outputFileName = "meta"
import os
outputFilePath = os.path.join(".", outputFileName)
# Write the metadata to a CSV
print('Writing metadata to CSV: %s' % outputFileName+'.csv')
with open(outputFilePath+'.csv', 'w') as outputStream:
    csvWriter = csv.DictWriter(outputStream, metadataFields)
    csvWriter.writeheader()
    for imageDetail in imageDetails:
        rowDict = imageDetail['meta']['clinical'].copy()
        rowDict['isic_id'] = imageDetail['name']
        csvWriter.writerow(rowDict)