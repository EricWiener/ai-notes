# YOLOv1

url: https://colab.research.google.com/drive/1P8032MYjqZJ-LtB0nP-19ndPncoCuHXb?authuser=1
![[AI-Notes/Implementations/yolov1-srcs/Untitled.png]]
Two-stage object detectors, such as ResNet, first propose regions where objects are likely, and then give per-class scores for those regions. YOLO on the other hand, produces the regions and gives per-class scores in one go.

## Grid Generator

You first divide the image up into an $S \times S$ grid. In our implementation, we use $S = 7$, so we have a $7 \times 7$ grid. We will place $A$ anchors centered at each of these centers. Below, the red dots show the equally-distributed center points.

![[The red dots show the center of each grid cell.]]

The red dots show the center of each grid cell.

## Anchor Generator

Now that we know where all the center points are, we can generate anchors centered at all these points. We choose the size of the anchors before hand by performing k-means clustering on the bounding boxes in our training data. This provides us with the ideal sized anchors. The anchor sizes are a hyperparameter.

```cpp
# We choose what we want the anchors to be
anchor_list = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]], **to_float_cuda)
```

We can now generate $A$ anchors ($A$ = `len(anchor_list)`) for each grid-cell. **The center of an anchor overlaps the center of the grid cell.** 

We denote the anchor coordinates as ($x_{tl}^a, y_{tl}^a, x_{br}^a, y_{br}^a$), indicating the coordinates of the top-left corner and the bottom-right corner accordingly.

We can visualize all the anchors generated for the center point of the center grid cell:

![[The anchors generated for the center grid-cell. Here, $A = 9$.]]

The anchors generated for the center grid-cell. Here, $A = 9$.

![[All the grid cells. Each red dot will have 9 anchors generated for it.]]

All the grid cells. Each red dot will have 9 anchors generated for it.

- Implementation
    
    ```python
    def GenerateAnchor(anc, grid):
      """
      Anchor generator.
    
      Inputs:
      - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
        each point in the grid. anc[a] = (w, h) gives the width and height of the
        a'th anchor shape.
      - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
        center of each feature from the backbone feature map. This is the tensor
        returned from GenerateGrid.
      
      Outputs:
      - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
        anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
        centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
        boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
        and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
        corners of the box.
      """
      anchors = None
      ##############################################################################
      # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
      # generate all the anchor coordinates for each image. Support batch input.   #
      ##############################################################################
      # Replace "pass" statement with your code
      A, _ = anc.shape
      B, H, W, _ = grid.shape
    
      anchors = torch.zeros((B, A, H, W, 4), device=grid.device, dtype=grid.dtype)
    
      for b in range(B):
        for a in range(A):
          anchor_w, anchor_h = anc[a]
    
          for h in range(H):
            for w in range(W):
              # Center of the the anchor
              cx, cy = grid[b][h][w]
    
              anchors[b][a][h][w][0] = cx - anchor_w / 2.0 # x_tl
              anchors[b][a][h][w][1] = cy - anchor_h / 2.0 # y_tl
              anchors[b][a][h][w][2] = cx + anchor_w / 2.0 # x_br
              anchors[b][a][h][w][3] = cy + anchor_h / 2.0 # y_br
    
      # TODO/QUESTION: anchors is cpu and grid is gpu
      # QUESTION:
      # How do I make this work
      anchors.to(grid.device)
    
      ##############################################################################
      #                               END OF YOUR CODE                             #
      ##############################################################################
    
      return anchors
    ```
    

## Proposal Generator

If we only use anchors to propose object locations, we can only cover 9x7x7=441 regions in the image. It is likely an object will not fall exactly into one of these regions. Therefore, the model will be able to predict transformations that **convert anchor boxes into region proposals.**

Now, consider an anchor box with center, width and height $(x_c^a,y_c^a,w^a,h^a)$. The network will predict a **transformation** $(t^x, t^y, t^w, t^h)$; applying this transformation to the anchor yields a **region proposal** with center, width and height $(x_c^p,y_c^p,w^p,h^p)$.

For YOLO, we assume that $t^x$ and $t^y$ are both in the range $-0.5\leq t^x,t^y\leq 0.5$, while $t^w$ and $t^h$ are real numbers in the range $(-\infty, \infty)$. Then we have:

- $x_c^p = x_c^a + t^x$
- $y_c^p = y_c^a + t^y$
- $w^p = w_a exp(t^w)$
- $h^p = h_a exp(t^h)$
- Implementation
    
    ```python
    def GenerateProposal(anchors, offsets, method='YOLO'):
      """
      Proposal generator.
    
      Inputs:
      - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
        by the coordinates of their top-left and bottom-right corners.
      - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
        convert anchor boxes into region proposals. The transformation
        offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
        anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
        (-0.5, 0.5).
      - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
      
      Outputs:
      - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
        coordinates of their top-left and bottom-right corners. Applying the
        transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
        proposal proposals[b, a, h, w].
      
      """
      assert(method in ['YOLO', 'FasterRCNN'])
      proposals = None
      ##############################################################################
      # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
      # compute the proposal coordinates using the transformation formulas above.  #
      ##############################################################################
      # Replace "pass" statement with your code
      B, A, H, W, _ = anchors.shape
    
      proposals = torch.zeros((B, A, H, W, 4), device=anchors.device, dtype=anchors.dtype)
    
      for b in range(B):
        for a in range(A):
          for h in range(H):
            for w in range(W):
              x_tl, y_tl, x_br, y_br = anchors[b][a][h][w]
    
              # Find the original center, width, and height
              a_w = x_br - x_tl
              a_h = y_br - y_tl
              cx = x_tl + a_w / 2
              cy = y_tl + a_h / 2
    
              tx, ty, tw, th = offsets[b][a][h][w]
    
              # The center shift depends on the method
              if method == 'YOLO':
                cx += tx
                cy += ty
              else:
                cx += tx * a_w
                cy += ty * a_h
    
              a_w *= torch.exp(tw)
              a_h *= torch.exp(th)
    
              proposals[b][a][h][w][0] = cx - a_w / 2 # x_tl
              proposals[b][a][h][w][1] = cy - a_h / 2 # y_tl
              proposals[b][a][h][w][2] = cx + a_w / 2 # x_br
              proposals[b][a][h][w][3] = cy + a_h / 2 # y_br
      ##############################################################################
      #                               END OF YOUR CODE                             #
      ##############################################################################
    
      return proposals
    ```
    

We can visualize what happens if we apply a transformation to the original red anchor box to end up with the proposal in green.

![[YOLOv1/Untitled 4.png]]

## Prediction Network

The prediction network will output a 3D grid that contains the transformations to the anchor boxes, the confidences for each anchor box, and the per-class scores for each cell. We don't flatten this output and pass it to a fully connected layer. Instead, we take advantage of its three dimensions to assign meaning to each of the cells.

The diagram below helps explain what the output of the prediction network tells you:

![[YOLOv1/Untitled 5.png]]

We already have:

- A grid generator that tells us where the centerpoints of the anchors should be.
- An anchor generator that gives us $A$ anchors of different sizes per grid centerpoint.
- A proposal generator that modifies the anchors generated by the anchor generator according to transformations.

For each of the anchors given by the anchor generator, the prediction network will tell us:

- How confident we are that this anchor contains an object.
- The $x, y, w, z$ transformation to apply to this anchor (we give the proposal generator these values).
- Scores for every cell in the $S \times S$ grid for each class.

The output of the prediction network is of dimension $(5 * A + C, S, S)$. The first $A * 5$ layers will contain (confidence, $t^x, t^y, t^w, t^h$) repeated $A$ times for each of the $S \times S$ cells. Then, the last $C$ layers will have the class score for each of the $S \times S$ cells.

### Prediction Network: Forward Pass

During the forward pass of the prediction network, the model will receive image features extracted by a CNN. It will then have different outputs depending on if it is training or testing.

- Implementation
    
    ```python
    class PredictionNetwork(nn.Module):
      def __init__(self, in_dim, hidden_dim=128, num_bboxes=2, num_classes=20, drop_ratio=0.3):
        super().__init__()
    
        assert(num_classes != 0 and num_bboxes != 0)
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes
    
        #++++++++++++++++++++++++++++++++++++++++++++++#
        #++++++++++++++++++++++++++++++++++++++++++++++#
        # ============== YOUR CODE HERE ============== #
    
        # TODO: 
        # Set up a network that will predict outputs for all bounding boxes. This     
        # network should have a 1x1 convolution with hidden_dim filters, followed    
        # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       
        # finally another 1x1 convolution layer to predict all outputs. You can      
        # use an nn.Sequential for this network, and store it in a member variable.  
        # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                
        # 5 predictions are in order (x, y, w, h, conf_score)                        
        # A = self.num_bboxes and C = self.num_classes.                                 
        self.model = nn.Sequential(
           nn.Conv2d(in_dim, hidden_dim, 1),
           nn.Dropout2d(drop_ratio),
           nn.LeakyReLU(),
           nn.Conv2d(hidden_dim, 5 * self.num_bboxes + self.num_classes, 1),
        )
      
        # ============== END OF CODE ================= # 
        #++++++++++++++++++++++++++++++++++++++++++++++#
        #++++++++++++++++++++++++++++++++++++++++++++++#
    ```
    

**Prediction Network: Forward Pass**

The forward pass of the model will output:

- `conf_scores`: Tensor of shape (B, A, H, W) giving predicted classification scores for all anchors. These are how confident we are for each anchor.
- `offsets`: Tensor of shape (B, A, 4, H, W) giving predicted transformations all all anchors.
- `class_scores`: Tensor of shape (B, C, H, W) giving classification scores for each spatial position.

The confidence, offsets, and class scores are returned for all possible anchors. 

- Implementation
    
    ```python
    def forward(self, features):
        """
        Run the forward pass of the network to predict outputs given features
        from the backbone network.
    
        Inputs:
        - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
          by the backbone network.
    
        Outputs:
        - bbox_xywh: Tensor of shape (B, A, 4, H, W) giving predicted offsets for 
          all bounding boxes.
        - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
          scores for all bounding boxes.
        - cls_scores: Tensor of shape (B, C, H, W) giving classification scores for
          each spatial position.
        """
        bbox_xywh, conf_scores, cls_scores = None, None, None
    
        # Use backbone features to predict bbox_xywh (offsets), conf_scores, and         
        # class_scores. Make sure conf_scores is between 0 and 1 by squashing the 
        # network output with a sigmoid. Also make sure the first two elements t^x 
        # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  
        # and subtracting 0.5. Also make sure, the last two elements of bbox_xywh
        # w and h are normlaized, i.e. squashed with a sigmoid between 0 and 1.
        # In the 5A+C dimension, the first 5*A would be bounding box offsets, 
        # and next C will be class scores.      
        A = self.num_bboxes
        C = self.num_classes
        B, _, H, W = features.shape
        
        # You first need to pass the features through the model to extract scores
        # (B, 5A + C, H, W)
        scores = self.model(features)
    
        # Get the class scores for every grid cell. These don't depend on the anchors,
        # so there is no A term.
        # cls_scores: (B, C, H, W)
        cls_scores = scores[:, -C:]
    
        # Get the anchor info (confidence and transformations)
        # Get (B, 5A, H, W) from (B, 5A + C, H, W)
        anchor_info = scores[:, 0:-C]
        
        # Reshape the anchor info so it is easier to manipulate
        # Reshape to (B, A, 5, H, W)
        anchor_info = anchor_info.reshape(B, A, 5, H, W)
    
        # Extract the confidence scores. By not specifying :-1, we 
        # get the desired shape (B, A, H, W) instead of (B, A, 1, H, W)
        # conf_scores: (B, A, H, W)
        conf_scores = anchor_info[:, :, -1]
    
        # Apply sigmoid to the conf_scores. conf_scores: (B, A, H, W)
        conf_scores = torch.sigmoid(conf_scores)
    
        # Get just the offsets
        # Get offsets of size (B, A, 4, H, W)
        bbox_xywh = anchor_info[:, :, :-1]  
    
        # Apply a sigmoid to the transformations
        bbox_xywh = torch.sigmoid(bbox_xywh)
    
        # Make tx and ty between -0.5 and 0.5 (avoid doing an in-place operation)
        # Need to create sub seperately to avoid warning:
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        sub = torch.zeros_like(bbox_xywh)
        sub[:, :, 0:2] = -0.5
        bbox_xywh = bbox_xywh + sub
    
        return bbox_xywh, conf_scores, cls_scores
    ```
    

**Prediction Network: Forward Pass (Training)**

If the model is used during training, we already know what the real bounding boxes are. We will divide the predicted bounding boxes into three types:

- Positive: these are bounding boxes that have a large overlap with the ground-truth bounding boxes.
- Negative: these are bounding boxes that don't overlap with the ground-truth bounding boxes. These are used as instances of the background class.
- Neutral: these are bounding boxes that partially overlap the ground-truth bounding boxes, but not enough to be considered positive. We ignore these because they are neither positive nor negative.

If we are in training, we want to be able to compare the predictions with the ground-truth values. This allows us to calculate the loss. Therefore, we want the following values:

- `conf_scores`: Tensor of shape (2*M, 1) giving the predicted classification scores for positive anchors and negative anchors (in that order).
    - We have the ground truth for the confidences for the positives anchors (100%)
    - And we have the ground truth for the negative anchors (0%)
- `offsets`: Tensor of shape (M, 4) giving predicted transformation for positive anchors.
    - We just have bounding boxes for the positive anchors, so we only care about the transformations for the positive anchors.
- `class_scores`: Tensor of shape (M, C) giving classification scores for positive anchors.
    - We only have class scores for the positive anchors, so we only care about these.

## Object Detector

We can now combine everything into a `SingleStageDetector` class that will perform object detection.

```python
class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self):
    raise NotImplementedError
  
  def inference(self):
    raise NotImplementedError
```

- Implementation
    
    ```python
    def forward(self, images, bboxes):
        """
        Training-time forward pass for the single-stage detector.
    
        Inputs:
        - images: Input images, of shape (B, 3, 224, 224)
        - bboxes: GT bounding boxes of shape (B, N, 5) (padded)
    
        Outputs:
        - total_loss: Torch scalar giving the total loss for the batch.
        """
        # weights to multiple to each loss term
        w_conf = 1 # for conf_scores
        w_reg = 1 # for offsets
        w_cls = 1 # for class_prob
    
        total_loss = None
        
        #++++++++++++++++++++++++++++++++++++++++++++++#
        #++++++++++++++++++++++++++++++++++++++++++++++#
        # ============== YOUR CODE HERE ============== #
        B, N, _ = bboxes.shape
        device = images.device    
    
        # Please refer to the above cell for an outline of key steps to impelement.                     
        """
        i)   Image feature extraction using Detector Backbone Network  
        """
        features = self.feat_extractor(images)
    
        """
        ii)  Grid List generation using Grid Generator  
        """
        grid_list = GenerateGrid(B, device=device)
    
        """
        iii) Compute offsets, conf_scores, cls_scores through the Prediction Network.  
        """
        # Offsets is (B, A, 4, H, W)
        # all_conf_scores is (B, A, H, W)
        # class scores is (B, C, H, W)
        offsets, all_conf_scores, class_scores = self.pred_network(features)
    
        """
        iv)  Calculate the proposals or actual bounding boxes using offsets stored in 
            offsets and grid_list by passing these into the GenerateProposal function 
            you wrote earlier.  
        """
        # Offsets is currently (B, A, 4, H, W). GenerateProposal expects (B, A, H, W, 4)
        offsets_reshaped = offsets.permute(0, 1, 3, 4, 2)
        proposals = GenerateProposal(grid_list, offsets_reshaped)
    
        """
        v)   Compute IoU between grid_list and proposals and then determine activated
            negative bounding boxes/proposals, and GT_conf_scores, GT_offsets, GT_class using the ReferenceOnActivatedBboxes function.
        """
        iou_mat = IoU(proposals, bboxes)
    
        # activated_anc_ind, negative_anc_ind are shape (M,) where M is number activated bounding boxes
        # GT_class also shape (M,)
        # GT_offsets is (M, 4)
        # GT_conf_scores is (M,)
        activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
          activated_anc_coord, negative_anc_coord = ReferenceOnActivatedBboxes(proposals, bboxes, grid_list, iou_mat, pos_thresh=0.7, neg_thresh=0.2)
    
        """
        vi)  The loss function for BboxRegression which expects the parametrization as   
            (x, y, sqrt(w), sqrt(h)) for the offsets and the GT_offsets also have  
            sqrt(w), sqrt(h) in the offsets as part of GT_offsets. So before the next step,   
            convert the bbox_xywh parametrization form (x, y, w, h) to (x, y, sqrt(w), sqrt(h)).   
        """
        # Offsets is (B, A, 4, H, W). We want to take the sqrt of w and h
        offsets[:, :, 2:4] = torch.sqrt(offsets[:, :, 2:4]) 
    
        """
        vii) Extract the confidence scores corresponding to the positive and negative   
            activated bounding box indices, classification scores for positive box indices,   
            offsets using positive box indices.  HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedBboxes.
            HINT: You can use the provided helper methods self._extract_bbox_data and   
            self._extract_class_scores to extract information for positive and   
            negative bounding boxes / proposals specified by activated_anc_ind and negative_anc_ind.
        """
        # Extract bounding box confidence scores for positive + negative
        # all_conf_scores is (B, A, H, W). Want it to be  (B, A, 1, H, W)
        all_conf_scores = all_conf_scores.unsqueeze(2)
        pos_conf_scores = self._extract_bbox_data(all_conf_scores, activated_anc_ind) # (M, 1)
        neg_conf_scores = self._extract_bbox_data(all_conf_scores, negative_anc_ind) # (M, 1)
        conf_scores = torch.cat((pos_conf_scores, neg_conf_scores), dim=0) # (2*M, 1)
    
        conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores) # scalar
    
        # Extract classification scores for positive
        # _extract_class_scores expects class_scores to be (B, C, H, W). class_scores already is 
        # Output is (M, C)
        positive_classification_scores = self._extract_class_scores(class_scores, activated_anc_ind)
    
        # Extract offsets just for positive bounding boxes
        # Offsets is (B, A, 4, H, W)
        # It should be (B, A, D, H, W)
        positive_offsets = self._extract_bbox_data(offsets, activated_anc_ind)
    
        """
        viii) Compute the total_loss which is formulated as:   
              total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,   
              where conf_loss is determined by ConfScoreRegression, w_reg by BboxRegression,  
              and w_cls by ObjectClassification. 
        """
        reg_loss = BboxRegression(positive_offsets, GT_offsets)
    
        # positive_classification_scores is (M, C)
        # GT_class is (M,)
        # batch_size is B
        # anc_per_img is A anchors per image * feature width * feature height
        B, A, _, H, W = offsets.shape
        anc_per_image = A * H * W
        cls_loss = ObjectClassification(positive_classification_scores, GT_class, B, anc_per_image, activated_anc_ind)
        total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss
    
        # ============== END OF CODE ================= # 
        #++++++++++++++++++++++++++++++++++++++++++++++#
        #++++++++++++++++++++++++++++++++++++++++++++++#   
      
        print('(weighted) conf loss: {:.4f}, reg loss: {:.4f}, cls loss: {:.4f}'.format(conf_loss, reg_loss, cls_loss))
    
        return total_loss
    ```
    

We can now implement the forward pass of our detector for training-time. The steps are as follows:

1. Image feature extraction using Detector Backbone Network: extract image features using the CNN.
    
    ```python
    # features is of shape (D, 7, 7)
    features = self.feat_extractor(images)
    ```
    
2. Grid List generation using Grid Generator: the grid gives us the center points. Each center point will receive $A$ anchors.
    
    ```python
    # Take the 7x7 grid of features (with depth D) and get anchors centered at each of the cells
    grid = GenerateGrid(B, device=device)
    ```
    
3. Compute `offsets`, `conf_scores`, `cls_scores` through the Prediction Network.
4. The anchor generation will create $A$ anchors per center point in the grid.
5. Compute IoU between anchors and GT boxes and then determine activated
negative anchors, and GT_conf_scores, GT_offsets, GT_class,
6. Compute conf_scores, offsets, class_prob through the prediction network
7. Compute the total_loss which is formulated as total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss, where conf_loss is determined by ConfScoreRegression, w_reg by BboxRegression, and w_cls by ObjectClassification.

## Object Detector Inference

The inference step is similar to the training step, except instead of calculating the loss on the proposals that match the ground-truth bounding boxes, we now no longer have ground-truth bounding boxes. Now, we need to threshold all the predicted proposals and decide what will constitute an actual object.

- We do this by first thresholding with a minimum proposal confidence. If the proposal has a higher confidence than the cut-off, it will move onto the next step.
- The next step is to perform non-maximum suppression (NMS) to handle overlapping bounding boxes.  This will ensure we don't have a lot of proposals all corresponding to the same object.

**Outputs:** we will end up keeping * proposals. This value will change depending on the thresholds we choose and the particular image.

- `final_propsals`: Proposals to keep after confidence score thresholding and NMS, a list of B (*x4) tensors
- `final_conf_scores`: Corresponding confidence scores, a list of B (*x1) tensors
- `final_class`: Corresponding class predictions, a list of B  (*x1) tensors

Note: `torchvision.ops.nms` doesn't work with batches, so we need to loop over the batch-index.

- Implementation
    
    ```python
    def detector_inference(self, images, thresh=0.5, nms_thresh=0.7):
        """"
        Inference-time forward pass for the single stage detector.
    
        Inputs:
        - images: Input images
        - thresh: Threshold value on confidence scores
        - nms_thresh: Threshold value on NMS
    
        Outputs:
        - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                          a list of B (*x4) tensors
        - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
        - final_class: Corresponding class predictions, a list of B  (*x1) tensors
        """
        final_proposals, final_conf_scores, final_class = [], [], []
        ##############################################################################
        # TODO: Predicting the final proposal coordinates `final_proposals`,         #
        # confidence scores `final_conf_scores`, and the class index `final_class`.  #
        # The overall steps are similar to the forward pass but now you do not need  #
        # to decide the activated nor negative anchors.                              #
        # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
        # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
        # threshold `nms_thresh`.                                                    #
        # The class index is determined by the class with the maximal probability.   #
        # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
        # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
        ##############################################################################
        # Replace "pass" statement with your code
        B = images.shape[0]
        device = images.device
    
    		# Don't want to calculate gradient during inference
        with torch.no_grad():
          # i) Image feature extraction,                                               #
          # features is of shaoe (D, 7, 7)
          features = self.feat_extractor(images)
    
          # ii) Grid and anchor generation.                                             #
          # Take the 7x7 grid of features (with depth D) and get anchors centered at each of the cells
          grid = GenerateGrid(B, device=device)
    
          # anchors is (B, A, H', W', 4)
          anchors = GenerateAnchor(self.anchor_list, grid) # all
    
          # =========== We can't do positive/negative anchors during test time ======= #
          # iii) Compute IoU between anchors and GT boxes and then determine activated/#
          #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
          # iou_mat = IoU(anchors, bboxes)
          # activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
          #  activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(anchors, bboxes, grid, iou_mat, pos_thresh=0.7, neg_thresh=0.2, method='YOLO')
          # =========== We can't do positive/negative anchors during test time ======= 
    
          # iv) Compute conf_scores, offsets, class_prob through the prediction network#
          # Outputs (During inference):
          # - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification scores for all anchors.
          # - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations all all anchors.
          # - class_scores: Tensor of shape (B, C, H, W) giving classification scores for each spatial position.
          conf_scores, offsets, class_scores = self.pred_network(features, pos_anchor_idx=None, neg_anchor_idx=None)
          B, C, H, W = class_scores.shape
    
          # We now need to re-arrange everything so it works nicely with creating the final lists
          # the final lists have length B.
          # final_proposals is a list of length B with *x4 tensors
          # final_conf_scores is a list of length B with *x1 tensors
          # final_class is a list of length B with *x1 tensors
          
          # conf_scores is (B, A, H, W). We want it to be (B, A*H*W) so it plays nicely
          # when we get the final lists. Now it is (B, A*H*W)
          conf_scores = conf_scores.reshape(B, -1)
    
          # offsets needs to be (B, A, H', W', 4) for GenerateProposal
          # Need to permute from (B, A, 4, H, W) -> (B, A, H', W', 4)
          reshaped_offsets = offsets.permute(0, 1, 3, 4, 2)
    
          # Proposals is of shape (B, A, H', W', 4)
          # anchors needs to be (B, A, H', W', 4)
          # reshaped_offsets is (B, A, H', W', 4)
          proposals = GenerateProposal(anchors, reshaped_offsets, method='YOLO')
          _, A, _, _, _ = proposals.shape
    
          # Now we need to reshape proposals so it plays nicely when we get the final lists
          # It is now (B, A*H'*W', 4). final_proposals will be *x4
          proposals = proposals.reshape(B, -1, 4)
    
          # Get the highest class score per feature cell. Originally (B, C, H, W)
          # Get max and make it (B, 1, H, W)
          # We permute and reshape before taking the max so that both 
          # highest_class_score and highest_class_idx are the correct shape of
          # (B, H*W). If we reshaped afterward, then we would have to reshape
          # both of them.
          # Now, for every item in the batch, it has a highest class score and index
          class_scores = class_scores.permute(0, 2, 3, 1).reshape(B, -1, C)
          highest_class_score, highest_class_idx = class_scores.max(dim=-1)
    
          # Need to apply the thresholding seperately on each image in the batch
          # We need to do this seperately because torchvision.ops.nms doesn't take
    			# in batches
          for b in range(B):
            # Threshold based on the conf_scores
            """
            torch.nonzero:
              If input has n dimensions, then the resulting indices tensor out is of size (z×n), 
              where z is the total number of non-zero elements in the input tensor.
            """
            # conf_scores is (B, A*H*W)
            # conf_scores[b] is (A*H*W)
            # conf_mask is (*, 1) since there is just 1 dimension to index (A*H*W dimension)
            # flatten this, so it becomes (*)
            conf_mask = torch.nonzero((conf_scores[b] > thresh)).squeeze(1)
    
            # Can't store these variables into the original conf_scores because then the
            # data for the next training instance will be overwritten. Need to create new variables
            instance_conf_scores = conf_scores[b, conf_mask] # of size (*)
    
            # Mask the proposals. Originally (B, A*H'*W', 4)
            # Now of dimension (*, 4)
            instance_proposals = proposals[b, conf_mask]
    
            # highest_class_idx is (B, H*W). conf_mask has indices for (B, A*H*W)
            # We no longer care about what anchor it belonged to. We just care about
            # the prediction, so we divide by A, so we get the indicies in terms of 
            # H and W.
            instance_highest_class_idx = highest_class_idx[b, conf_mask // A]
    
            # Threshold using nms_thresh
            kept_indices = torchvision.ops.nms(instance_proposals, instance_conf_scores, nms_thresh)
    
            # final_proposals is a list of length B with *x4 tensors
            final_proposals.append(instance_proposals[kept_indices])
    
            # final_conf_scores is a list of length B with *x1 tensors
            final_conf_scores.append(instance_conf_scores[kept_indices].reshape(-1, 1))
    
            # final_class is a list of length B with *x1 tensors
            final_class.append(instance_highest_class_idx[kept_indices].reshape(-1, 1))
    
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return final_proposals, final_conf_scores, final_class
      SingleStageDetector.inference = detector_inference
    ```