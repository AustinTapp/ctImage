import SimpleITK as sitk

# Load the segmentation image
seg = sitk.ReadImage('segmentation.nii.gz')

# Perform boundary refinement
seg_filtered = sitk.SmoothingRecursiveGaussian(seg, sigma=1.0)

# Perform edge detection using the Canny filter
seg_edges = sitk.CannyEdgeDetection(seg_filtered, lowerThreshold=1.0, upperThreshold=10.0)

# Apply a binary threshold to the edge image to obtain a mask
mask = sitk.BinaryThreshold(seg_edges, lowerThreshold=0.0, upperThreshold=1.0)

# Apply the mask to the original segmentation image to create a new, refined segmentation
refined_seg = sitk.Mask(seg, mask)

# Save the refined segmentation to a file
sitk.WriteImage(refined_seg, 'refined_segmentation.nii.gz')

#-----------------

import SimpleITK as sitk

# Load the original segmentation image
orig_seg = sitk.ReadImage('original_segmentation.nii.gz')

# Load the refined segmentation image
refined_seg = sitk.ReadImage('refined_segmentation.nii.gz')

# Apply label fusion to combine the original and refined segmentations
fusion = sitk.LabelVoting(orig_seg, refined_seg)

# Save the fused segmentation to a file
sitk.WriteImage(fusion, 'fused_segmentation.nii.gz')
