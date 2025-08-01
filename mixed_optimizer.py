"""
混合优化器模块
支持使用 SparseAdam 优化 embedding 参数，AdamW 优化其他参数
用于大规模embedding的内存优化
"""

import torch
from torch.optim import SparseAdam, AdamW
try:
    from adamw_bf16 import AdamWBF16
except ImportError:
    # 如果adamw_bf16不存在，使用标准AdamW作为备用
    AdamWBF16 = AdamW
    
try:
    from loguru import logger
except ImportError:
    # 如果loguru不存在，使用标准print函数
    class Logger:
        def info(self, msg):
            print(f"[INFO] {msg}")
    logger = Logger()


class MixedOptimizer:
    """
    混合优化器类，使用 SparseAdam 优化 embedding 参数，AdamW 优化其他参数
    这样可以显著减少大规模embedding训练时的显存占用
    
    特点：
    - 自动检测模型中设置为 sparse=True 的 embedding 层
    - 对稀疏 embedding 使用 SparseAdam 优化器（显存友好）
    - 对其他参数使用 AdamW 优化器（支持权重衰减）
    """
    
    def __init__(self, model, sparse_lr=1e-3, dense_lr=1e-3, weight_decay=0.01, 
                 adam_betas=(0.9, 0.999), use_bf16=False, use_sparse_optimizer=True):
        """
        初始化混合优化器
        
        参数:
            model: 模型实例
            sparse_lr: 稀疏参数（embedding）学习率
            dense_lr: 密集参数学习率
            weight_decay: 密集参数的权重衰减
            adam_betas: Adam优化器的beta参数
            use_bf16: 是否使用bfloat16
            use_sparse_optimizer: 是否使用稀疏优化器（如果为False，所有参数使用AdamW）
        """
        self.use_sparse_optimizer = use_sparse_optimizer
        self.sparse_lr = sparse_lr
        self.dense_lr = dense_lr
        self.weight_decay = weight_decay
        self.adam_betas = adam_betas
        self.use_bf16 = use_bf16
        
        if use_sparse_optimizer:
            # 分离embedding参数和其他参数
            sparse_params, dense_params = self._separate_parameters(model)
            
            # 创建稀疏优化器（用于embedding参数）
            if sparse_params:
                self.sparse_optimizer = SparseAdam(sparse_params, lr=sparse_lr, betas=adam_betas)
                logger.info(f"创建SparseAdam优化器: {len(sparse_params)} 个稀疏参数, lr={sparse_lr}")
            else:
                self.sparse_optimizer = None
                logger.info("没有找到embedding参数")
            
            # 创建密集优化器（用于其他参数）
            if dense_params:
                opt_cls = AdamWBF16 if use_bf16 else AdamW
                self.dense_optimizer = opt_cls(
                    dense_params, 
                    lr=dense_lr, 
                    betas=adam_betas, 
                    weight_decay=weight_decay
                )
                logger.info(f"创建{opt_cls.__name__}优化器: {len(dense_params)} 个密集参数, lr={dense_lr}")
            else:
                self.dense_optimizer = None
                logger.info("没有找到密集参数")
        else:
            # 使用传统的单一优化器
            opt_cls = AdamWBF16 if use_bf16 else AdamW
            self.dense_optimizer = opt_cls(
                model.parameters(),
                lr=dense_lr,
                betas=adam_betas,
                weight_decay=weight_decay,
            )
            self.sparse_optimizer = None
            logger.info(f"创建传统{opt_cls.__name__}优化器: lr={dense_lr}")
    
    def _separate_parameters(self, model):
        """
        动态识别并分离稀疏embedding参数和其他参数
        
        返回:
            (sparse_params, dense_params): 稀疏参数列表和密集参数列表
        """
        sparse_params = []
        dense_params = []
        
        # 调试：打印所有模型参数名
        logger.info("=== 所有模型参数名 ===")
        for name, param in model.named_parameters():
            logger.info(f"参数: {name}, 形状: {param.shape}, requires_grad: {param.requires_grad}")
        logger.info("=== 参数名列表结束 ===")
        
        # 动态检测稀疏embedding层
        sparse_embedding_names = self._detect_sparse_embeddings(model)
        logger.info(f"检测到的稀疏embedding层: {sparse_embedding_names}")
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # 检查参数是否属于稀疏embedding层
            is_sparse = False
            for sparse_name in sparse_embedding_names:
                if name.startswith(sparse_name) and name.endswith('.weight'):
                    is_sparse = True
                    break
            
            if is_sparse:
                sparse_params.append(param)
                logger.info(f"✅ 稀疏参数: {name}, 形状: {param.shape}")
            else:
                dense_params.append(param)
                logger.info(f"📌 密集参数: {name}, 形状: {param.shape}")
        
        logger.info(f"参数分离完成: {len(sparse_params)} 个稀疏参数, {len(dense_params)} 个密集参数")
        return sparse_params, dense_params
    
    def _detect_sparse_embeddings(self, model):
        """
        动态检测模型中所有设置为sparse=True的embedding层
        
        参数:
            model: PyTorch模型
            
        返回:
            list: 稀疏embedding层的名称列表
        """
        sparse_embedding_names = []
        
        # 检查所有模块
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                # 检查embedding是否设置为稀疏
                if hasattr(module, 'sparse') and module.sparse:
                    sparse_embedding_names.append(name)
                    logger.info(f"🎯 发现稀疏embedding层: {name}")
        
        return sparse_embedding_names
    
    def _is_sparse_param(self, param_name):
        """
        判断参数是否为稀疏参数（已弃用，现在使用动态检测）
        
        参数:
            param_name: 参数名称
            
        返回:
            bool: 是否为稀疏参数
        """
        # 这个方法已被 _detect_sparse_embeddings 替代
        # 保留是为了向后兼容
        logger.warning("_is_sparse_param 方法已弃用，使用 _detect_sparse_embeddings 替代")
        return False
    
    def zero_grad(self):
        """清零所有优化器的梯度"""
        if self.sparse_optimizer:
            self.sparse_optimizer.zero_grad()
        if self.dense_optimizer:
            self.dense_optimizer.zero_grad()
    
    def step(self):
        """执行优化步骤"""
        if self.sparse_optimizer:
            self.sparse_optimizer.step()
        if self.dense_optimizer:
            self.dense_optimizer.step()
    
    def state_dict(self):
        """返回优化器状态字典"""
        state = {}
        if self.sparse_optimizer:
            state['sparse'] = self.sparse_optimizer.state_dict()
        if self.dense_optimizer:
            state['dense'] = self.dense_optimizer.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """加载优化器状态字典"""
        if 'sparse' in state_dict and self.sparse_optimizer:
            self.sparse_optimizer.load_state_dict(state_dict['sparse'])
            logger.info("加载稀疏优化器状态")
        if 'dense' in state_dict and self.dense_optimizer:
            self.dense_optimizer.load_state_dict(state_dict['dense'])
            logger.info("加载密集优化器状态")
    
    def parameters(self):
        """返回所有参数（兼容性方法）"""
        params = []
        if self.sparse_optimizer:
            for group in self.sparse_optimizer.param_groups:
                params.extend(group['params'])
        if self.dense_optimizer:
            for group in self.dense_optimizer.param_groups:
                params.extend(group['params'])
        return params
    
    @property
    def param_groups(self):
        """返回所有参数组（兼容性属性）"""
        groups = []
        if self.sparse_optimizer:
            groups.extend(self.sparse_optimizer.param_groups)
        if self.dense_optimizer:
            groups.extend(self.dense_optimizer.param_groups)
        return groups