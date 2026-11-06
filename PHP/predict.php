<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');

function sendJson(int $status, array $payload): never
{
    http_response_code($status);
    echo json_encode($payload, JSON_UNESCAPED_SLASHES);
    exit;
}

function iniSizeBytes(string $value): int
{
    $value = trim($value);
    if ($value === '') {
        return 0;
    }

    $unit = strtolower(substr($value, -1));
    $size = (float) $value;
    return (int) match ($unit) {
        'g' => $size * 1024 * 1024 * 1024,
        'm' => $size * 1024 * 1024,
        'k' => $size * 1024,
        default => $size,
    };
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    sendJson(405, ['error' => 'Use POST to submit a video.']);
}

$postLimit = iniSizeBytes((string) ini_get('post_max_size'));
if (
    empty($_FILES)
    && $postLimit > 0
    && (int) ($_SERVER['CONTENT_LENGTH'] ?? 0) > $postLimit
) {
    sendJson(413, [
        'error' => 'The request exceeds PHP post_max_size. Set post_max_size to at least 210M.',
    ]);
}

$upload = $_FILES['video'] ?? null;
if (!is_array($upload)) {
    sendJson(400, ['error' => 'Choose a video to test.']);
}
$uploadError = (int) ($upload['error'] ?? UPLOAD_ERR_NO_FILE);
if ($uploadError !== UPLOAD_ERR_OK) {
    if ($uploadError === UPLOAD_ERR_INI_SIZE || $uploadError === UPLOAD_ERR_FORM_SIZE) {
        sendJson(413, [
            'error' => 'Video exceeds PHP upload_max_filesize. Set it to at least 200M.',
        ]);
    }
    sendJson(400, ['error' => 'The video upload did not complete.']);
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
    error_log('Video prediction failed: video model folder is unavailable.');
    sendJson(500, ['error' => 'Translation is temporarily unavailable. Please try again later.']);
}

$temporaryBase = tempnam(sys_get_temp_dir(), 'aslt_');
if ($temporaryBase === false) {
    error_log('Video prediction failed: could not create a temporary upload file.');
    sendJson(500, ['error' => 'We could not process this video. Please try again later.']);
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
        $diagnostic = trim((string) $stderr);
        if ($diagnostic !== '') {
            error_log('Video prediction process failed: ' . $diagnostic);
        }
        throw new RuntimeException('Video prediction process failed.');
    }

    $decoded = json_decode((string) $stdout, true, 512, JSON_THROW_ON_ERROR);
    if (!is_array($decoded) || empty($decoded['predictions'])) {
        throw new RuntimeException('The model returned no predictions.');
    }
    $response = $decoded;
    $status = 200;
} catch (Throwable $error) {
    error_log('Video prediction failed: ' . $error->getMessage());
    $response = [
        'error' => 'We could not translate this video. Please try recording it again or choose another supported video.',
    ];
} finally {
    if (is_file($temporaryBase)) {
        unlink($temporaryBase);
    }
    if (is_file($temporaryVideo)) {
        unlink($temporaryVideo);
    }
}

sendJson($status, $response);