<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');

function sendJson(int $status, array $payload): never
{
    http_response_code($status);
    echo json_encode($payload, JSON_UNESCAPED_SLASHES);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    sendJson(405, ['error' => 'Use POST to submit a video.']);
}

$upload = $_FILES['video'] ?? null;
if (!is_array($upload)) {
    sendJson(400, ['error' => 'Choose a video to test.']);
}
if ($upload['error'] !== UPLOAD_ERR_OK) {
    $message = $upload['error'] === UPLOAD_ERR_INI_SIZE || $upload['error'] === UPLOAD_ERR_FORM_SIZE
        ? 'Video exceeds the PHP upload limit.'
        : 'The video upload did not complete.';
    sendJson(400, ['error' => $message]);
}

$extension = strtolower(pathinfo((string) $upload['name'], PATHINFO_EXTENSION));
$allowedExtensions = ['avi', 'm4v', 'mkv', 'mov', 'mp4', 'webm'];
if (!in_array($extension, $allowedExtensions, true)) {
    sendJson(400, ['error' => 'Use a supported video file.']);
}
if ((int) $upload['size'] > 200 * 1024 * 1024) {
    sendJson(413, ['error' => 'Video exceeds the 200 MB limit.']);
}

$model = (string) ($_POST['model'] ?? 'cnn3d');
if (!in_array($model, ['cnn_gru', 'cnn3d'], true)) {
    sendJson(400, ['error' => 'Choose an available model.']);
}

$projectRoot = realpath(dirname(__DIR__) . DIRECTORY_SEPARATOR . 'videoModel');
if ($projectRoot === false) {
    sendJson(500, ['error' => 'The video model folder is unavailable.']);
}

$temporaryBase = tempnam(sys_get_temp_dir(), 'aslt_');
if ($temporaryBase === false) {
    sendJson(500, ['error' => 'Could not create a temporary upload file.']);
}
$temporaryVideo = $temporaryBase . '.' . $extension;
$status = 500;
$response = ['error' => 'Prediction failed.'];

try {
    if (!rename($temporaryBase, $temporaryVideo)) {
        throw new RuntimeException('Could not prepare the temporary video file.');
    }
    if (!move_uploaded_file((string) $upload['tmp_name'], $temporaryVideo)) {
        throw new RuntimeException('Could not save the uploaded video.');
    }

    $python = getenv('PYTHON_EXECUTABLE') ?: 'python';
    $command = [
        $python,
        '-m',
        'src.predict_video',
        $temporaryVideo,
        '--model',
        $model,
        '--json',
    ];
    $descriptors = [
        0 => ['pipe', 'r'],
        1 => ['pipe', 'w'],
        2 => ['pipe', 'w'],
    ];
    $process = proc_open($command, $descriptors, $pipes, $projectRoot);
    if (!is_resource($process)) {
        throw new RuntimeException('Could not start Python.');
    }

    fclose($pipes[0]);
    $stdout = stream_get_contents($pipes[1]);
    $stderr = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    $exitCode = proc_close($process);

    if ($exitCode !== 0) {
        error_log(trim((string) $stderr));
        throw new RuntimeException(trim((string) $stderr) ?: 'The model could not read this video.');
    }

    $decoded = json_decode((string) $stdout, true, 512, JSON_THROW_ON_ERROR);
    if (!is_array($decoded) || empty($decoded['predictions'])) {
        throw new RuntimeException('The model returned no predictions.');
    }
    $response = $decoded;
    $status = 200;
} catch (Throwable $error) {
    error_log($error->getMessage());
    $response = ['error' => $error->getMessage()];
} finally {
    if (is_file($temporaryBase)) {
        unlink($temporaryBase);
    }
    if (is_file($temporaryVideo)) {
        unlink($temporaryVideo);
    }
}

sendJson($status, $response);