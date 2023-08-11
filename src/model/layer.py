"""
This module is derived from Catalyst contrib layer.
https://github.com/catalyst-team/catalyst/tree/master/catalyst/contrib/layers
https://catalyst-team.github.io/catalyst/api/contrib.html#layers
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):
    """Implementation of
    `CosFace\: Large Margin Cosine Loss for Deep Face Recognition`_.

    .. _CosFace\: Large Margin Cosine Loss for Deep Face Recognition:
        https://arxiv.org/abs/1801.09414

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature.
            Default: ``64.0``.
        m: margin.
            Default: ``0.35``.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.

    Example:
        >>> layer = CosFaceLoss(5, 10, s=1.31, m=0.1)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> self.engine.backward(loss)

    """

    def __init__(  # noqa: D107
        self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.35
    ):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "CosFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m}"
            ")"
        )
        return rep

    def forward(
        self, input: torch.Tensor, target: torch.LongTensor = None
    ) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.

        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes
            (out_features).
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m

        if target is None:
            return cosine

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits


class AdaCos(nn.Module):
    """Implementation of
    `AdaCos\: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations`_.

    .. _AdaCos\: Adaptively Scaling Cosine Logits for\
        Effectively Learning Deep Face Representations:
        https://arxiv.org/abs/1905.00292

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        dynamical_s: option to use dynamical scale parameter.
            If ``False`` then will be used initial scale.
            Default: ``True``.
        eps: operation accuracy.
            Default: ``1e-6``.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.

    Example:
        >>> layer = AdaCos(5, 10)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> self.engine.backward(loss)

    """  # noqa: E501,W505

    def __init__(  # noqa: D107
        self,
        in_features: int,
        out_features: int,
        dynamical_s: bool = True,
        eps: float = 1e-6,
    ):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = math.sqrt(2) * math.log(out_features - 1)
        self.eps = eps

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "AdaCos("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"eps={self.eps}"
            ")"
        )
        return rep

    def forward(
        self, input: torch.Tensor, target: torch.LongTensor = None
    ) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.

        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes
            (out_features).
        """
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        if target is None:
            return cos_theta

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        if self.train:
            with torch.no_grad():
                b_avg = (
                    torch.where(
                        one_hot < 1,
                        torch.exp(self.s * cos_theta),
                        torch.zeros_like(cos_theta),
                    )
                    .sum(1)
                    .mean()
                )
                theta_median = theta[one_hot > 0].median()
                theta_median = torch.min(
                    torch.full_like(theta_median, math.pi / 4), theta_median
                )
                self.s = (torch.log(b_avg) / torch.cos(theta_median)).item()

        logits = self.s * cos_theta
        return logits


class AMSoftmax(nn.Module):
    """Implementation of
    `AMSoftmax: Additive Margin Softmax for Face Verification`_.
    .. _AMSoftmax\: Additive Margin Softmax for Face Verification:
        https://arxiv.org/pdf/1801.05599.pdf
    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature.
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
        eps: operation accuracy.
            Default: ``1e-6``.
    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.
    Example:
        >>> layer = AMSoftmax(5, 10, s=1.31, m=0.5)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> self.engine.backward(loss)
    """

    def __init__(  # noqa: D107
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        eps: float = 1e-6,
    ):
        super(AMSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.eps = eps

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "ArcFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m},"
            f"eps={self.eps}"
            ")"
        )
        return rep

    def forward(
        self, input: torch.Tensor, target: torch.LongTensor = None
    ) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.
        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes
            (out_features).
        """
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))

        if target is None:
            return cos_theta

        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        logits = torch.where(one_hot.bool(), cos_theta - self.m, cos_theta)
        logits *= self.s

        return logits


class ArcFace(nn.Module):
    """Implementation of
    `ArcFace: Additive Angular Margin Loss for Deep Face Recognition`_.
    .. _ArcFace\: Additive Angular Margin Loss for Deep Face Recognition:
        https://arxiv.org/abs/1801.07698v1
    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature.
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
        eps: operation accuracy.
            Default: ``1e-6``.
    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.
    Example:
        >>> layer = ArcFace(5, 10, s=1.31, m=0.5)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> self.engine.backward(loss)
    """

    def __init__(  # noqa: D107
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        eps: float = 1e-6,
    ):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.threshold = math.pi - m
        self.eps = eps

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "ArcFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m},"
            f"eps={self.eps}"
            ")"
        )
        return rep

    def forward(
        self, input: torch.Tensor, target: torch.LongTensor = None
    ) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.
        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes
            (out_features).
        """
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))

        if target is None:
            return cos_theta

        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        mask = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot)

        logits = torch.cos(torch.where(mask.bool(), theta + self.m, theta))
        logits *= self.s

        return logits


class SubCenterArcFace(nn.Module):
    """Implementation of
    `Sub-center ArcFace: Boosting Face Recognition
    by Large-scale Noisy Web Faces`_.
    .. _Sub-center ArcFace\: Boosting Face Recognition \
        by Large-scale Noisy Web Faces:
        https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature,
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
        k: number of possible class centroids.
            Default: ``3``.
        eps (float, optional): operation accuracy.
            Default: ``1e-6``.
    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.
    Example:
        >>> layer = SubCenterArcFace(5, 10, s=1.31, m=0.35, k=2)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> self.engine.backward(loss)
    """

    def __init__(  # noqa: D107
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        k: int = 3,
        eps: float = 1e-6,
    ):
        super(SubCenterArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        self.k = k
        self.eps = eps

        self.weight = nn.Parameter(torch.FloatTensor(k, in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        self.threshold = math.pi - self.m

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "SubCenterArcFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m},"
            f"k={self.k},"
            f"eps={self.eps}"
            ")"
        )
        return rep

    def forward(
        self, input: torch.Tensor, target: torch.LongTensor = None
    ) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.
        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes.
        """
        feats = F.normalize(input).unsqueeze(0).expand(self.k, *input.shape)  # k*b*f
        wght = F.normalize(self.weight, dim=1)  # k*f*c
        cos_theta = torch.bmm(feats, wght)  # k*b*f
        cos_theta = torch.max(cos_theta, dim=0)[0]  # b*f
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        if target is None:
            return cos_theta

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        selected = torch.where(
            theta > self.threshold, torch.zeros_like(one_hot), one_hot
        )

        logits = torch.cos(torch.where(selected.bool(), theta + self.m, theta))
        logits *= self.s

        return logits


class ArcMarginProduct(nn.Module):
    """Implementation of Arc Margin Product.
    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.
    Example:
        >>> layer = ArcMarginProduct(5, 10)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding)
        >>> loss = loss_fn(output, target)
        >>> self.engine.backward(loss)
    """

    def __init__(self, in_features: int, out_features: int):  # noqa: D107
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "ArcMarginProduct("
            f"in_features={self.in_features},"
            f"out_features={self.out_features}"
            ")"
        )
        return rep

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes
            (out_features).
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine
    

class CurricularFace(nn.Module):
    """Implementation of
    `CurricularFace: Adaptive Curriculum Learning\
        Loss for Deep Face Recognition`_.
    .. _CurricularFace\: Adaptive Curriculum Learning\
        Loss for Deep Face Recognition:
        https://arxiv.org/abs/2004.00288
    Official `pytorch implementation`_.
    .. _pytorch implementation:
        https://github.com/HuangYG123/CurricularFace
    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature.
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.
    Example:
        >>> layer = CurricularFace(5, 10, s=1.31, m=0.5)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> self.engine.backward(loss)
    """  # noqa: RST215

    def __init__(  # noqa: D107
        self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.5
    ):
        super(CurricularFace, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer("t", torch.zeros(1))

        nn.init.normal_(self.weight, std=0.01)

    def __repr__(self) -> str:  # noqa: D105
        rep = (
            "CurricularFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"m={self.m},s={self.s}"
            ")"
        )
        return rep

    def forward(
        self, input: torch.Tensor, label: torch.LongTensor = None
    ) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            label: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.
        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes.
        """
        cos_theta = torch.mm(F.normalize(input), F.normalize(self.weight, dim=0))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        if label is None:
            return cos_theta

        target_logit = cos_theta[torch.arange(0, input.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = (
            target_logit * self.cos_m - sin_theta * self.sin_m
        )  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(
            target_logit > self.threshold, cos_theta_m, target_logit - self.mm
        )

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t

        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s

        return output    
    

def get_layer(
    layer_name: str,
    embedding_size: int,
    num_classes: int,
    s: float,
    m: float,
    eps: float,
    k: int,
    ) -> torch.nn.Module:
    if layer_name == 'AdaCos':
        return AdaCos(
            in_features=embedding_size, 
            out_features=num_classes,
            eps=eps,
        )
    elif layer_name == 'AMSoftmax':
        return AMSoftmax(
            in_features=embedding_size, 
            out_features=num_classes,
            s=s,
            m=m,
            eps=eps,
        )
    elif layer_name == 'ArcFace':
        return ArcFace(
            in_features=embedding_size, 
            out_features=num_classes,
            s=s,
            m=m,
            eps=eps,
        )
    elif layer_name == 'CosFace':
        return CosFace(
            in_features=embedding_size, 
            out_features=num_classes,
            s=s,
            m=m,
        )
    elif layer_name == 'CurricularFace':
        return CurricularFace(
            in_features=embedding_size, 
            out_features=num_classes,
            s=s,
            m=m,
        )
    elif layer_name == 'SubCenterArcFace':
        return SubCenterArcFace(
            in_features=embedding_size, 
            out_features=num_classes,
            s=s,
            m=m,
            k=k,
            eps=eps,
        )
    else:
        raise ValueError(f'Unknown optimizer: {layer_name}')