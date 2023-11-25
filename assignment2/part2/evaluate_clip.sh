echo "CIFAR10 on train set:"
python3 clipzs.py --dataset cifar10 --split train

echo "CIFAR10 on test set:"
python3 clipzs.py --dataset cifar10 --split test

echo "CIFAR100 on train set:"
python3 clipzs.py --dataset cifar100 --split train

echo "CIFAR100 on train set:"
python3 clipzs.py --dataset cifar100 --split test

#echo "Zero-shot prompting for dominant image color:"
#python3 clipzs.py --prompt_template "{}" --class_names red green blue --visualize_predictions
#python3 clipzs.py --prompt_template "{0} {0} {0}" --class_names red green blue --visualize_predictions
#python3 clipzs.py --prompt_template "{0}" --class_names red_redish_pink_cherry green_greenish_olive_grassy blue_bluish_azure_navy --visualize_predictions
#python3 clipzs.py --prompt_template "The dominant color of the image is {}" --class_names red green blue --visualize_predictions
#python3 clipzs.py --prompt_template "The image is {}" --class_names redish greenish blueish --visualize_predictions

# echo "Zero-shot prompting for natural vs man-made objects:"
# python3 clipzs.py --prompt_template "The main object on the image is {}" --class_names a_man-made_machine occurs_in_nature --visualize_predictions
