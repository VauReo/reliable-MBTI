from runners.common import load_config_from_args


def main() -> None:
    _, config = load_config_from_args('Generate SFT data from teacher traces.')
    print(f"[generate_sft_data] experiment={config['experiment_name']}")
    print(f"[generate_sft_data] student_model_id={config['model']['student_model_id']}")
    print('[generate_sft_data] TODO: implement teacher prompting and rejection sampling.')


if __name__ == '__main__':
    main()
