import vtk
import itk
import DataProcessing
import SimpleITK as sitk
import ModelConfiguration

# process example CT image

ctImage = sitk.ReadImage("CT.nii.gz")
binaryImage = DataProcessing.CreateBoneMask(ctImage)
ctImage = DataProcessing.ResampleAndMaskImage(ctImage, binaryImage)
#sitk.WriteImage(ctImage, "toModel.mha")

# model
modelPath = './MiccaiFinalModel.dat'
device = ModelConfiguration.getDevice()
model = ModelConfiguration.adaptModel(modelPath, device)
imageData = ModelConfiguration.adaptData(ctImage, device)

landmarks, all_seven_labels = ModelConfiguration.runModel(model, ctImage, binaryImage, imageData)

sitk.WriteImage(all_seven_labels, 'LabeledCT.mha')

point_writer = vtk.vtkXMLPolyDataWriter()
point_writer.SetFileName('CTLPoints.vtp')
point_writer.SetInputData(landmarks)
point_writer.Write()
