import vtk
import itk
import DataProcessing
import SimpleITK as sitk
import ModelConfiguration

# process example CT image

ctImage = sitk.ReadImage("D:\\Data\\CNH_Paired\\Ready\\ReorientedCT\\15_CT_RO.nii.gz")
binaryImage = DataProcessing.CreateBoneMask(ctImage)
ctImage = DataProcessing.ResampleAndMaskImage(ctImage, binaryImage)
#sitk.WriteImage(ctImage, "test/54/54MRcropToCT_toModel.mha")

# model
modelPath = './MiccaiFinalModel.dat'
device = ModelConfiguration.getDevice()
model = ModelConfiguration.adaptModel(modelPath, device)
imageData = ModelConfiguration.adaptData(ctImage, device)

landmarks, all_seven_labels = ModelConfiguration.runModel(model, ctImage, binaryImage, imageData)

sitk.WriteImage(all_seven_labels, 'D:\\Data\\CNH_Paired\\15_T1labeled.mha')

point_writer = vtk.vtkXMLPolyDataWriter()
point_writer.SetFileName("D:\\Data\\CNH_Paired\\3_T1CTLPoints.vtp")
point_writer.SetInputData(landmarks)
point_writer.Write()
