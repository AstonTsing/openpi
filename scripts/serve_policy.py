import dataclasses  # 导入 dataclasses，用于把配置参数类声明成数据类，自动生成初始化等方法。
import enum  # 导入 enum，用于定义环境类型枚举 EnvMode。
import logging  # 导入 logging，用于输出 policy server 启动等运行日志。
import socket  # 导入 socket，用于获取当前机器 hostname 和内网 IP，方便日志中确认服务位置。

import tyro  # 导入 tyro，用于把 dataclass 参数结构自动转换成命令行参数解析器。

from openpi.policies import policy as _policy  # 导入 policy 模块并命名为 _policy，用于类型标注和可选的推理记录包装器。
from openpi.policies import policy_config as _policy_config  # 导入 policy_config，用于根据训练配置和 checkpoint 创建可推理的 policy。
from openpi.serving import websocket_policy_server  # 导入 WebSocket policy server，用于把 policy 暴露成网络推理服务。
from openpi.training import config as _config  # 导入训练配置模块，用于按配置名读取 pi05_libero、pi05_droid 等配置。


class EnvMode(enum.Enum):  # 定义支持的默认环境枚举，命令行可通过 --env 选择其中一种环境。
    """Supported environments."""  # 说明这个枚举列出了脚本支持的环境类型。

    ALOHA = "aloha"  # ALOHA 真实机器人环境，对应默认 ALOHA policy/checkpoint。
    ALOHA_SIM = "aloha_sim"  # ALOHA 仿真环境，对应默认 ALOHA simulation policy/checkpoint。
    DROID = "droid"  # DROID 机器人环境，对应默认 pi05_droid policy/checkpoint。
    LIBERO = "libero"  ############################################# LIBERO 仿真 benchmark 环境，对应默认 pi05_libero policy/checkpoint。


@dataclasses.dataclass  # 将 Checkpoint 声明为数据类，便于 tyro 解析 policy:checkpoint 子命令参数。
class Checkpoint:  # 定义“从指定 checkpoint 加载 policy”的参数结构。
    """Load a policy from a trained checkpoint."""  # 说明这个参数分支用于从训练好的 checkpoint 加载模型。

    # Training config name (e.g., "pi0_aloha_sim").
    config: str  # 训练配置名，例如 pi05_libero 或 pi05_droid，用来决定模型结构和数据变换。
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str  # checkpoint 目录，可以是本地路径，也可以是 gs://openpi-assets/... 这样的远程路径。


@dataclasses.dataclass  # 将 Default 声明为数据类，便于 tyro 解析默认 policy 参数分支。
class Default:  # 定义“按环境使用默认 checkpoint”的参数结构。
    """Use the default policy for the given environment."""  # 说明这个分支会根据 env 自动选择默认 policy/checkpoint。


@dataclasses.dataclass  # 将 Args 声明为数据类，tyro 会据此生成整个脚本的命令行接口。
class Args:  # 定义 serve_policy.py 脚本的所有命令行参数。
    """Arguments for the serve_policy script."""  # 说明这个数据类保存脚本参数。

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM  # 默认环境是 ALOHA_SIM；只有使用默认 policy 分支时才会用这个值。

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None  # 可选默认语言指令；当输入 observation 没有 prompt 时注入给模型。

    # Port to serve the policy on.
    port: int = 8000  # policy server 监听端口，客户端会通过这个端口请求动作。
    # Record the policy's behavior for debugging.
    record: bool = False  # 是否记录 policy 输入输出，打开后会把推理轨迹写到 policy_records 目录。

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)  # policy 加载方式；默认按 env 自动选择，也可通过 policy:checkpoint 指定。


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {  # 定义每种默认环境对应的训练配置名和 checkpoint 路径。
    EnvMode.ALOHA: Checkpoint(  # ALOHA 环境默认使用 pi05_aloha 配置。
        config="pi05_aloha",  # ALOHA 默认加载 pi05_aloha 训练/推理配置。
        dir="gs://openpi-assets/checkpoints/pi05_base",  # ALOHA 默认 checkpoint 指向 pi05_base。
    ),  # 结束 ALOHA 默认 checkpoint 配置。
    EnvMode.ALOHA_SIM: Checkpoint(  # ALOHA_SIM 环境默认使用 pi0_aloha_sim 配置。
        config="pi0_aloha_sim",  # ALOHA 仿真默认加载 pi0_aloha_sim 配置。
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",  # ALOHA 仿真默认 checkpoint 路径。
    ),  # 结束 ALOHA_SIM 默认 checkpoint 配置。
    EnvMode.DROID: Checkpoint(  # DROID 环境默认使用 pi05_droid 配置。
        config="pi05_droid",  # DROID 默认加载 pi05_droid 配置，也就是 π0.5-DROID。
        dir="gs://openpi-assets/checkpoints/pi05_droid",  # DROID 默认 checkpoint 指向官方 pi05_droid。
    ),  # 结束 DROID 默认 checkpoint 配置。
    EnvMode.LIBERO: Checkpoint(  # LIBERO 环境默认使用 pi05_libero 配置。
        config="pi05_libero",  # LIBERO 默认加载 pi05_libero 配置，也就是 π0.5-LIBERO。
        dir="gs://openpi-assets/checkpoints/pi05_libero",  # LIBERO 默认 checkpoint 指向官方 pi05_libero。
    ),  # 结束 LIBERO 默认 checkpoint 配置。
}  # 结束默认环境到 checkpoint 的映射表。


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:  # 根据环境枚举创建默认 policy。
    """Create a default policy for the given environment."""  # 说明这个函数会按 env 选择 DEFAULT_CHECKPOINT 并创建 policy。
    if checkpoint := DEFAULT_CHECKPOINT.get(env):  # 从默认映射表里取出当前环境对应的 checkpoint；取到才进入分支。
        return _policy_config.create_trained_policy(  # 根据训练配置和 checkpoint 目录构造可推理的 Policy 对象。
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt  # 读取配置名对应的 TrainConfig，并传入 checkpoint 路径和默认 prompt。
        )  # 返回创建好的默认 policy。
    raise ValueError(f"Unsupported environment mode: {env}")  # 如果 env 不在默认映射表中，就抛出不支持该环境的错误。


def create_policy(args: Args) -> _policy.Policy:  # 根据命令行参数创建最终要服务的 policy。
    """Create a policy from the given arguments."""  # 说明这个函数会根据 args.policy 选择加载逻辑。
    match args.policy:  # 根据 policy 参数分支做模式匹配：指定 checkpoint 或使用默认环境。
        case Checkpoint():  # 如果命令行使用 policy:checkpoint，就进入显式 checkpoint 加载流程。
            return _policy_config.create_trained_policy(  # 用用户指定的配置名和 checkpoint 目录创建 policy。
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt  # 读取用户指定配置，并传入用户指定 checkpoint 路径和默认 prompt。
            )  # 返回从指定 checkpoint 创建的 policy。
        case Default():  # 如果命令行没有指定 checkpoint，就按 env 使用默认 checkpoint。
            return create_default_policy(args.env, default_prompt=args.default_prompt)  # 调用默认 policy 创建函数，并传入 env 和默认 prompt。


def main(args: Args) -> None:  # 脚本主函数，接收 tyro 解析出来的 Args 参数。
    policy = create_policy(args)  # 根据命令行参数加载模型并创建 policy。
    policy_metadata = policy.metadata  # 取出 policy 元数据，稍后传给 websocket server 供客户端读取。

    # Record the policy's behavior.
    if args.record:  # 如果用户打开 --record，就包装 policy 以记录推理输入输出。
        policy = _policy.PolicyRecorder(policy, "policy_records")  # 用 PolicyRecorder 包装原 policy，并把记录写到 policy_records 目录。

    hostname = socket.gethostname()  # 获取当前服务器的主机名，用于日志提示。
    local_ip = socket.gethostbyname(hostname)  # 根据主机名解析当前服务器 IP，用于日志提示客户端连接地址。
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)  # 打印即将创建服务的主机名和 IP。

    server = websocket_policy_server.WebsocketPolicyServer(  # 创建 WebSocket policy server，把 policy 暴露成网络服务。
        policy=policy,  # 传入要被服务化的 policy，server 收到 observation 后会调用它推理 actions。
        host="0.0.0.0",  # 监听所有网卡地址，使同机器和局域网内其他机器都可能连接。
        port=args.port,  # 使用命令行指定的端口，默认是 8000。
        metadata=policy_metadata,  # 把 policy 元数据传给 server，方便客户端了解 policy 相关信息。
    )  # 完成 WebSocketPolicyServer 对象创建。
    server.serve_forever()  # 启动服务并持续阻塞运行，等待客户端发送 observation 并返回 actions。


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)  # 初始化 logging，让 INFO 级别日志能输出到终端。
    main(tyro.cli(Args))  # 用 tyro 从命令行解析 Args，然后调用 main 启动 policy server。
