"""Additive dataset/task/style/factorial embeddings."""

from __future__ import annotations

from torch import Tensor, nn

from cfmusic.conditioning.schema import ConditionBatch


class AdditiveConditionEmbedding(nn.Module):
    def __init__(
        self,
        *,
        num_datasets: int,
        num_tasks: int,
        num_styles: int,
        num_genres: int,
        num_emotions: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.dataset = nn.Embedding(num_datasets, embedding_dim)
        self.task = nn.Embedding(num_tasks, embedding_dim)
        self.style = nn.Embedding(num_styles, embedding_dim)
        self.genre = nn.Embedding(num_genres, embedding_dim)
        self.emotion = nn.Embedding(num_emotions, embedding_dim)

    def forward(self, condition: ConditionBatch) -> Tensor:
        context = self.dataset(condition.dataset_id) + self.task(condition.task_id)
        semantic = self.style(condition.style_id)
        embedding = context + semantic
        if condition.genre_id is not None:
            genre = self.genre(condition.genre_id)
        else:
            # Keep every embedding parameter in the static DDP graph without changing
            # the unconditional value. This removes find_unused_parameters traversal.
            genre = self.genre.weight[0].sum() * 0.0
        embedding = embedding + genre
        semantic = semantic + genre
        if condition.emotion_id is not None:
            emotion = self.emotion(condition.emotion_id)
        else:
            emotion = self.emotion.weight[0].sum() * 0.0
        embedding = embedding + emotion
        if condition.condition_mask is None:
            return embedding
        semantic = semantic + emotion
        mask = condition.condition_mask.to(semantic).reshape(-1, 1)
        if mask.shape[0] != semantic.shape[0]:
            raise ValueError("Condition mask batch does not match embedded conditions")
        return context + semantic * mask
