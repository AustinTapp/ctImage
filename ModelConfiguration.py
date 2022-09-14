import torch
import numpy as np
import vtk
import SimpleITK as sitk

def getDevice(device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    return(device)

def adaptData(ctImage, device):
    """
    Data for final model needs to be in batch x channel x height x width x depth format.
    """
    imageData = sitk.GetArrayFromImage(ctImage)
    if len(imageData.shape)==3:
        imageData = np.expand_dims(imageData,axis=0)
        imageData = np.expand_dims(imageData,axis=0)
    elif len(imageData.shape)==4:
        imageData = np.expand_dims(imageData,axis=0)

    return(torch.tensor(imageData, device = device, dtype = torch.float32))

def landmarkPrediction(heatmap, ctImage):

    landmarks = vtk.vtkPolyData()
    landmarks.SetPoints(vtk.vtkPoints())
    origin = ctImage.GetOrigin()
    spacing = ctImage.GetSpacing()
    for p in range(4):

        coords = np.flip(np.where(heatmap[0].cpu().detach().numpy()[p]==heatmap[0].cpu().detach().numpy()[p].max()))

        coords = origin + spacing * coords.ravel()

        landmarks.GetPoints().InsertNextPoint(coords[0], coords[1], coords[2])

    return(landmarks)

def boneLabeling(segmentation, ctImage, binaryImage):
    bonelabels = segmentation.cpu().detach().numpy()
    bonelabels = np.argmax(bonelabels[0, :, :, : ,:], axis=0).astype(np.int16)

    similarityTransform = sitk.AffineTransform(3)
    similarityTransform.SetIdentity()

    bonelabels = sitk.GetImageFromArray(bonelabels.astype(np.uint16))
    bonelabels.CopyInformation(ctImage)
    bonelabels = sitk.Resample(bonelabels, binaryImage, similarityTransform, sitk.sitkNearestNeighbor)

    labelsArray = sitk.GetArrayFromImage(bonelabels)
    labelsArray[sitk.GetArrayViewFromImage(binaryImage)<= 0] = 0
    bonelabels = sitk.GetImageFromArray(labelsArray)
    bonelabels.CopyInformation(binaryImage)

    return(bonelabels)

def seven_labels(segmentation, ctImage, binaryImage):
    bonelabels = segmentation.cpu().detach().numpy()
    bonelabels = np.argmax(bonelabels[0, :, :, : ,:], axis=0).astype(np.int16)

    # copy labels for 'subtraction' image
    bonelabels_converted = np.copy(bonelabels)
    bonelabels_converted[bonelabels_converted != 0] = 1

    # copy labels again for coordinates used in bone specific segmentation
    RF = np.copy(bonelabels)
    RF[RF != 1] = 0
    LF = np.copy(bonelabels)
    LF[LF != 2] = 0

    # reposition label to image space
    similarityTransform = sitk.AffineTransform(3)
    similarityTransform.SetIdentity()

    bonelabels = sitk.GetImageFromArray(bonelabels.astype(np.uint32))
    bonelabels.CopyInformation(ctImage)
    bonelabels = sitk.Resample(bonelabels, binaryImage, similarityTransform, sitk.sitkNearestNeighbor)

    boneArray = sitk.GetArrayFromImage(bonelabels)
    boneArray[sitk.GetArrayViewFromImage(binaryImage)<= 0] = 0
    bonelabels = sitk.GetImageFromArray(boneArray)
    bonelabels.CopyInformation(binaryImage)

    # reposition repeated for RF
    RF = sitk.GetImageFromArray(RF.astype(np.uint16))
    RF.CopyInformation(ctImage)
    RF = sitk.Resample(RF, binaryImage, similarityTransform, sitk.sitkNearestNeighbor)

    RFArray = sitk.GetArrayFromImage(RF)
    RFArray[sitk.GetArrayViewFromImage(binaryImage) <= 0] = 0
    RF = sitk.GetImageFromArray(RFArray)
    RF.CopyInformation(binaryImage)

    # reposition repeated for LF
    LF = sitk.GetImageFromArray(LF.astype(np.uint16))
    LF.CopyInformation(ctImage)
    LF = sitk.Resample(LF, binaryImage, similarityTransform, sitk.sitkNearestNeighbor)

    LFArray = sitk.GetArrayFromImage(LF)
    LFArray[sitk.GetArrayViewFromImage(binaryImage) <= 0] = 0
    LF = sitk.GetImageFromArray(LFArray)
    LF.CopyInformation(binaryImage)

    # reposition the binary bone as well
    bonelabels_converted = sitk.GetImageFromArray(bonelabels_converted.astype(np.uint32))
    bonelabels_converted.CopyInformation(ctImage)
    bonelabels_converted = sitk.Resample(bonelabels_converted, binaryImage, similarityTransform, sitk.sitkNearestNeighbor)

    boneArray = sitk.GetArrayFromImage(bonelabels_converted)
    boneArray[sitk.GetArrayViewFromImage(binaryImage) <= 0] = 0
    bonelabels_converted = sitk.GetImageFromArray(boneArray)
    bonelabels_converted.CopyInformation(binaryImage)

    # subtract the original binary image by the bone label binary image
    subtract = sitk.SubtractImageFilter()
    subtracted = subtract.Execute(binaryImage, bonelabels_converted)
    subtracted = sitk.ConnectedComponent(subtracted)
    sorted_component_image = sitk.RelabelComponent(subtracted, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1

    # post-processing to clean things up
    fill_hole = sitk.BinaryFillhole(largest_component_binary_image)
    medianfilter = sitk.MedianImageFilter()
    medianfilter.SetRadius(1)
    medianfilter.Execute(fill_hole)
    size = fill_hole.GetSize()

    # determine the midline of the skull based on existing segmentations
    # calculate the center of the right frontal segmentation
    RF_label_statistic = sitk.LabelShapeStatisticsImageFilter()
    RF_label_statistic.Execute(RF == 1)
    RF_centroid = RF_label_statistic.GetCentroid(1)
    RF_centroid_coord = RF.TransformPhysicalPointToIndex(RF_centroid)

    # calculate the center of the left frontal segmentation
    LF_label_statistic = sitk.LabelShapeStatisticsImageFilter()
    LF_label_statistic.Execute(LF == 2)
    LF_centroid = LF_label_statistic.GetCentroid(1)
    LF_centroid_coord = LF.TransformPhysicalPointToIndex(LF_centroid)

    # use the centers to define the midpoint
    midpoint = ((RF_centroid_coord[0] + LF_centroid_coord[0])/2, (RF_centroid_coord[1] + LF_centroid_coord[1])/2, (RF_centroid_coord[2] + LF_centroid_coord[2])/2)
    crop_point = int(midpoint[0])

    # set the left side of the subtracted, postprocessed, whole binary image to 0 to maintain the right temporal bone
    right_temporal_array = sitk.GetArrayFromImage(fill_hole)
    right_temporal_array[:, :, crop_point:size[0]] = 0
    right_temporal_image = sitk.GetImageFromArray(right_temporal_array.astype(np.uint16))
    right_temporal_image.CopyInformation(fill_hole)

    # set the right side of the subtracted, postprocessed, whole binary image to 0 to maintain the left temporal bone
    left_temporal_array = sitk.GetArrayFromImage(fill_hole)
    left_temporal_array[:, :, 0:crop_point-1] = 0
    left_temporal_image = sitk.GetImageFromArray(left_temporal_array.astype(np.uint16))
    left_temporal_image.CopyInformation(fill_hole)

    # convert the binary images to LabelMaps
    right_temporal_mapping = sitk.BinaryImageToLabelMapFilter().Execute(right_temporal_image)
    left_temporal_mapping = sitk.BinaryImageToLabelMapFilter().Execute(left_temporal_image)

    # ensure all labels have only one value
    right_temporal_mapping = sitk.AggregateLabelMap(right_temporal_mapping)
    left_temporal_mapping = sitk.AggregateLabelMap(left_temporal_mapping)

    # convert the original bone labelmap from an image type to a label map type
    bonelabels_labelmap = sitk.LabelImageToLabelMap(bonelabels, 0)

    # combine the original 5 bone labelmap, the right temporal labelmap and the left temporal labelmap IN THAT ORDER
    combine_label_maps = sitk.MergeLabelMapFilter()
    combine_label_maps.SetMethod(2)

    # combine and return
    seven_bonelabels = combine_label_maps.Execute(bonelabels_labelmap, right_temporal_mapping, left_temporal_mapping)
    label_map_to_image = sitk.LabelMapToLabelImageFilter()
    seven_bones_image = label_map_to_image.Execute(seven_bonelabels)

    return seven_bones_image


def adaptModel(modelPath, device):
    model = torch.jit.load(modelPath)
    model.to(device=device) # Sending to device
    model.eval()   
    return(model)

def runModel(model, ctImage, binaryImage, imageData):
    heatmap_pred, segmentation_pred, _, _ = model(imageData)

    landmarks = landmarkPrediction(heatmap_pred, ctImage)
    seven_bonelabels = seven_labels(segmentation_pred, ctImage, binaryImage)

    return landmarks, seven_bonelabels
