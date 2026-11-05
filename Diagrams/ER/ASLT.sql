CREATE TABLE `users` (
  `user_id` int PRIMARY KEY AUTO_INCREMENT,
  `username` varchar(100) UNIQUE NOT NULL,
  `email` varchar(255) UNIQUE NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `full_name` varchar(150),
  `profile_photo` varchar(500),
  `role` varchar(20) NOT NULL DEFAULT 'user',
  `status` varchar(20) NOT NULL DEFAULT 'pending',
  `created_at` datetime,
  `updated_at` datetime
);

CREATE TABLE `otp` (
  `otp_id` int PRIMARY KEY AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `otp_code` varchar(10) NOT NULL,
  `purpose` varchar(30) NOT NULL,
  `expires_at` datetime NOT NULL,
  `is_verified` boolean DEFAULT false,
  `created_at` datetime
);

CREATE TABLE `login_history` (
  `login_id` int PRIMARY KEY AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `login_time` datetime,
  `login_status` varchar(20) NOT NULL,
  `device_info` varchar(255),
  `ip_address` varchar(45)
);

CREATE TABLE `user_sessions` (
  `session_id` int PRIMARY KEY AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `refresh_token_hash` varchar(255) UNIQUE NOT NULL,
  `device_name` varchar(150),
  `platform` varchar(30) NOT NULL,
  `created_at` datetime,
  `expires_at` datetime NOT NULL,
  `last_seen_at` datetime,
  `revoked_at` datetime
);

CREATE TABLE `translations` (
  `translation_id` int PRIMARY KEY AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `mode` varchar(30) NOT NULL,
  `input_type` varchar(30) NOT NULL,
  `input_file_path` varchar(500),
  `input_text` text,
  `predicted_text` text,
  `result_type` varchar(20),
  `confidence_score` decimal(5,2),
  `created_at` datetime
);

CREATE TABLE `model_accuracy` (
  `accuracy_id` int PRIMARY KEY AUTO_INCREMENT,
  `model_name` varchar(150) NOT NULL,
  `model_version` varchar(50),
  `total_samples` int,
  `correct_predictions` int,
  `accuracy_percentage` decimal(5,2),
  `evaluated_at` datetime
);

CREATE TABLE `sign_materials` (
  `material_id` int PRIMARY KEY AUTO_INCREMENT,
  `category` varchar(30) NOT NULL,
  `sign_label` varchar(100) NOT NULL,
  `image_path` varchar(500),
  `explanation` text,
  `created_at` datetime
);

CREATE TABLE `quizzes` (
  `quiz_id` int PRIMARY KEY AUTO_INCREMENT,
  `created_by` int NOT NULL,
  `title` varchar(150) NOT NULL,
  `description` text,
  `status` varchar(20) DEFAULT 'active',
  `created_at` datetime
);

CREATE TABLE `quiz_questions` (
  `question_id` int PRIMARY KEY AUTO_INCREMENT,
  `quiz_id` int NOT NULL,
  `question_text` text,
  `image_path` varchar(500),
  `correct_answer` varchar(255) NOT NULL,
  `options` json
);

CREATE TABLE `quiz_results` (
  `result_id` int PRIMARY KEY AUTO_INCREMENT,
  `quiz_id` int NOT NULL,
  `user_id` int NOT NULL,
  `score` int,
  `total_questions` int,
  `completed_at` datetime
);

CREATE TABLE `feedbacks` (
  `feedback_id` int PRIMARY KEY AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `message` text,
  `rating_score` int NOT NULL,
  `status` varchar(20) DEFAULT 'pending',
  `admin_reply` text,
  `created_at` datetime,
  `updated_at` datetime
);

CREATE TABLE `cursor_effects` (
  `effect_id` int PRIMARY KEY AUTO_INCREMENT,
  `effect_name` varchar(100) NOT NULL,
  `effect_path` varchar(500),
  `status` varchar(20) DEFAULT 'active',
  `created_at` datetime
);

CREATE TABLE `user_settings` (
  `setting_id` int PRIMARY KEY AUTO_INCREMENT,
  `user_id` int UNIQUE NOT NULL,
  `effect_id` int,
  `updated_at` datetime
);

CREATE TABLE `search_history` (
  `search_id` int PRIMARY KEY AUTO_INCREMENT,
  `admin_id` int NOT NULL,
  `search_keyword` varchar(255),
  `search_type` varchar(50),
  `searched_at` datetime
);

ALTER TABLE `otp` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`);

ALTER TABLE `login_history` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`);

ALTER TABLE `user_sessions` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`);

ALTER TABLE `translations` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`);

ALTER TABLE `quizzes` ADD FOREIGN KEY (`created_by`) REFERENCES `users` (`user_id`);

ALTER TABLE `quiz_questions` ADD FOREIGN KEY (`quiz_id`) REFERENCES `quizzes` (`quiz_id`);

ALTER TABLE `quiz_results` ADD FOREIGN KEY (`quiz_id`) REFERENCES `quizzes` (`quiz_id`);

ALTER TABLE `quiz_results` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`);

ALTER TABLE `feedbacks` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`);

ALTER TABLE `user_settings` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`);

ALTER TABLE `user_settings` ADD FOREIGN KEY (`effect_id`) REFERENCES `cursor_effects` (`effect_id`);

ALTER TABLE `search_history` ADD FOREIGN KEY (`admin_id`) REFERENCES `users` (`user_id`);
