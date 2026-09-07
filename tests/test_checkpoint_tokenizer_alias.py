from omegaconf import OmegaConf

from g05.utils.checkpoint.ckpt_utils import _apply_action_tokenizer_sidecar


def test_local_sidecar_preserves_tokenizer_alias_and_target(tmp_path):
    (tmp_path / 'action_tokenizer.pt').touch()
    cfg = OmegaConf.create({
        'tokenizer': {'_target_': 'Tokenizer', 'vq_config': {'ckpt_dir': 'old.pt', 'block_size': 8}},
        'model': {
            'tokenizer': '${tokenizer}',
            'model_arch': {'action_tokenizer': '${model.tokenizer._target_}',
                           'AT_CONFIG': '${model.tokenizer.vq_config}'},
        },
    })
    assert _apply_action_tokenizer_sidecar(cfg, tmp_path)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved['model']['model_arch']['action_tokenizer'] == 'Tokenizer'
    assert resolved['model']['tokenizer']['vq_config']['block_size'] == 8
    assert resolved['model']['model_arch']['AT_CONFIG']['ckpt_dir'] == str(tmp_path / 'action_tokenizer.pt')
