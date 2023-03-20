# Udacity_project4

## Step 1: 

Used an ml.t3.medium notebook as from my previous experiences with similar projects this instance seemed sufficient with a low cost 

Notebook set up: 
![alt text](https://github.com/archthegit/udacity_project4/blob/main/new_screenshots/Screenshot%202023-03-19%20at%2010.51.55%20PM.png)

S3 Bucket used: 
![alt text](https://github.com/archthegit/udacity_project4/blob/main/new_screenshots/Screenshot%202023-03-20%20at%205.13.41%20PM.png)

Deployed endpoint:
![alt text](https://github.com/archthegit/udacity_project4/blob/main/new_screenshots/Screenshot%202023-03-20%20at%202.58.16%20AM.png)

## Step 2: 
I chose the g5.2xlarge as it was on the list of approved instances that work with the Deep Learning AMI, I needed the pytorch env to be able to run the `solution.py` script. It was one of the cheaper options providing enough cpu and memory (32 GiB) to train the model. I played it safe and chose a model with more specs than needed since it only cost $1.212/hr and i would only be using it for 10-20 mins.

<img width="1273" alt="Screenshot 2023-03-20 at 5 27 13 PM" src="https://user-images.githubusercontent.com/22144490/226298612-39ecbb28-4eb5-4b03-a29a-0273b20be205.png">

EC2 instance with `model.pth` saved to `TrainedModels` directory
![alt text](https://github.com/archthegit/udacity_project4/blob/main/new_screenshots/Screenshot%202023-03-20%20at%201.21.35%20AM.png)

#### EC2 Code vs Sagemaker Code:
* Not a lot of hyperparameter tuning in the EC2 code compared to Sagemaker code
* Deployment method is different, one generates a model.pth file and one uses an endpoint that can be accessed publicly

## Step 3: 

I created a lambda function using the `lambdafunction.py` script where I replaced the `endpoint_Name` variable with my previously created endpoint `pytorch-inference-2023-03-19-18-54-03-855`. The lambda function simply calls this endpoint and returns the response. I created a test event to see if my function works correctly:

```
{ "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/20113314/Carolina-Dog-standing-outdoors.jpg" }
```

Result of lambda function test:
![alt text](https://github.com/archthegit/udacity_project4/blob/main/new_screenshots/Screenshot%202023-03-20%20at%204.16.41%20PM.png)

```
Test Event Name
test

Response
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "text/plain",
    "Access-Control-Allow-Origin": "*"
  },
  "type-result": "<class 'str'>",
  "COntent-Type-In": "LambdaContext([aws_request_id=09360260-4de3-4ea2-af2e-96d067e143fa,log_group_name=/aws/lambda/project4,log_stream_name=2023/03/20/[$LATEST]d6abca5676624089a21830d01fd426a1,function_name=project4,memory_limit_in_mb=128,function_version=$LATEST,invoked_function_arn=arn:aws:lambda:us-east-1:775241169951:function:project4,client_context=None,identity=CognitoIdentity([cognito_identity_id=None,cognito_identity_pool_id=None])])",
  "body": "[[-14.143568992614746, -5.684410095214844, -1.8749933242797852, 1.0599637031555176, 1.5907402038574219, -9.410658836364746, 3.0227255821228027, 1.385068416595459, -13.199945449829102, 0.2824665307998657, 1.7210664749145508, 0.4888707399368286, -9.935746192932129, 2.262202024459839, 2.4222753047943115, 3.347254991531372, -9.389200210571289, -0.5058841109275818, -0.33502423763275146, 1.7883074283599854, -7.682830333709717, -4.413837432861328, 1.0785170793533325, -6.095279693603516, -0.4689871072769165, -10.435538291931152, -2.356555938720703, 0.19481587409973145, -2.488048553466797, -4.760436534881592, -3.755894184112549, -4.093752861022949, -9.263851165771484, -1.754446268081665, -13.825780868530273, -8.840593338012695, -1.4469965696334839, -9.024213790893555, 1.1076340675354004, -2.542921781539917, -3.6020495891571045, -9.03349781036377, 2.61519718170166, -4.319530963897705, 3.4584975242614746, -7.591547966003418, -2.3079190254211426, -1.651167631149292, -2.8108091354370117, -1.700773000717163, -3.731165885925293, -10.32164192199707, -9.238126754760742, -1.5285320281982422, -7.632514953613281, 1.8710193634033203, -4.603858947753906, -11.991665840148926, -2.1461586952209473, -3.4940292835235596, -10.87259292602539, -4.850460529327393, -6.786985874176025, -4.215777397155762, 0.8865344524383545, -12.795674324035645, 1.1294605731964111, -9.566117286682129, -2.7429378032684326, -0.6227406859397888, -0.6780720353126526, -5.000589370727539, -5.843488693237305, -9.59928035736084, -7.8359222412109375, -3.028465986251831, -9.624259948730469, -2.729515790939331, -2.523822069168091, 2.3772106170654297, 0.8827787637710571, -9.296175956726074, 1.72808837890625, -1.4903112649917603, -2.0019474029541016, -11.3505220413208, -2.4974653720855713, -11.48370361328125, -5.013983249664307, -1.8143545389175415, -8.952452659606934, -10.935545921325684, -5.06876802444458, -6.129566192626953, 0.07544615864753723, -0.5298648476600647, -2.626542091369629, -10.328303337097168, -13.231423377990723, -11.008482933044434, -11.791641235351562, -0.5376484990119934, -3.4365146160125732, -6.607066631317139, -6.152707576751709, -13.245508193969727, -6.411252498626709, -1.62942373752594, -0.352207213640213, 1.6104347705841064, -6.683472156524658, -2.8108246326446533, -9.330506324768066, -8.079345703125, -7.919734954833984, 1.1410871744155884, -18.581933975219727, 1.747987151145935, -6.9517059326171875, -0.9426793456077576, -0.3137938380241394, -1.0914843082427979, -13.857415199279785, -3.4875152111053467, -13.333316802978516, -1.1660922765731812, -6.684042453765869, 2.0908584594726562, -8.438952445983887, -3.444429397583008, -9.801985740661621, -4.906007289886475, -13.57454776763916]]"
}

Function Logs
START RequestId: 09360260-4de3-4ea2-af2e-96d067e143fa Version: $LATEST
Context::: LambdaContext([aws_request_id=09360260-4de3-4ea2-af2e-96d067e143fa,log_group_name=/aws/lambda/project4,log_stream_name=2023/03/20/[$LATEST]d6abca5676624089a21830d01fd426a1,function_name=project4,memory_limit_in_mb=128,function_version=$LATEST,invoked_function_arn=arn:aws:lambda:us-east-1:775241169951:function:project4,client_context=None,identity=CognitoIdentity([cognito_identity_id=None,cognito_identity_pool_id=None])])
EventType:: <class 'dict'>
END RequestId: 09360260-4de3-4ea2-af2e-96d067e143fa
REPORT RequestId: 09360260-4de3-4ea2-af2e-96d067e143fa	Duration: 936.98 ms	Billed Duration: 937 ms	Memory Size: 128 MB	Max Memory Used: 73 MB

Request ID
09360260-4de3-4ea2-af2e-96d067e143fa
```

## Step 4:

Lambda function setup:

![alt text](https://github.com/archthegit/udacity_project4/blob/main/new_screenshots/Screenshot%202023-03-20%20at%204.19.47%20PM.png)

#### Security policy:

My policy has full SageMaker access, which is a lot more permissive than it needs to be. Also roles that are  old or inactive can be dangerous as people who no longer work on the project can have access to the resources and can jeopardize the project with malicious intent. Likewise, roles with policies for functions that the project is no longer using can be dangerous because users can use resources to modify other unrelated resources that might affect the success and security of the project. I can make it secure by reducing the scope to my specific endpoint by naming the resource and specifying only read actions on the endpoint. I believe my AWS Workspace is relatively secure but constant monitoring of security policies across EC2, SageMaker, IAM roles, Lambda functions is required to make it as restrictive as possible.

![alt text](https://github.com/archthegit/udacity_project4/blob/main/new_screenshots/Screenshot%202023-03-20%20at%209.29.11%20PM.png)

## Step 5:
 
#### Concurrency:

I set up a provisioned concurrency of 3 so that my Lambda function can handle that many more calls concurrently.

![alt text](https://github.com/archthegit/udacity_project4/blob/main/new_screenshots/Screenshot%202023-03-20%20at%205.06.44%20PM.png)

#### Auto-scaling

I set min instance to 1 and max instance to 5 so that it can scale accordingly. I also set the scale-in/out cool down to 60 seconds so that instances can scale in/out faster and it doesnt affect the availability of the endpoint.

![alt text](https://github.com/archthegit/udacity_project4/blob/main/new_screenshots/Screenshot%202023-03-20%20at%205.10.54%20PM.png)
