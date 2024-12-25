from typing import List


class Config:
  dataset_path: str
  story_path: str
  augmented_story_path: str
  label_path: str
  augmented_label_path: str
  labels: List[str]
  model_path: str
  epoch: int
  batch_size: int
  num_nn_models: int

  def __init__(
      self,
      dataset_path: str = "../datasets/",
      story_path: str = "../datasets/CBU0521DD_stories/",
      augmented_story_path: str = "../datasets/CBU0521DD_stories/_augmented/",
      label_path: str = "../datasets/CBU0521DD_stories_attributes.csv",
      augmented_label_path: str = "../datasets/CBU0521DD_stories_attributes_augmented.csv",
      labels: List[str] = ["True Story", "Deceptive Story"],
      model_path: str = "../models/",
      epoch: int = 100,
      batch_size: int = 10,
      num_nn_models:int = 3,
  ):
    self.dataset_path = dataset_path
    self.story_path = story_path
    self.augmented_story_path = augmented_story_path
    self.label_path = label_path
    self.augmented_label_path = augmented_label_path
    self.labels = labels
    self.model_path = model_path
    self.epoch = epoch
    self.batch_size = batch_size
    self.num_nn_models = num_nn_models
